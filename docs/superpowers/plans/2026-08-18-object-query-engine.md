# Object Query Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Name an object in a labelled room and get back its 3D bounding box, carved tight by cross-view agreement, with the option to promote the result into `objects.json` and absorb the duplicate entries it replaces.

**Architecture:** One primitive does all the work — project a 3D point into another frame and ask whether it lands inside that frame's cached 2D box. Pointed at one object's boxes it carves away clutter; pointed at two candidate objects it decides whether they are the same thing. Three new pure modules (`camera.py`, `consensus.py`, `query.py`) sit on top of the existing `Reconstruction` and `observations.json`; nothing in the reconstruction or fusion stages changes.

**Tech Stack:** Python 3.11, numpy<2, scipy (already declared), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-18-object-query-design.md`

## Global Constraints

- **Part 1 only.** This plan covers the geometry core, the query engine and the CLI verb. The HTTP endpoint, the viewer search box and the mask overlay UI are Part 2 and are explicitly out of scope.
- **No new dependencies.** `scipy` is already in `pyproject.toml`; nothing else may be added.
- **Every module in this plan is pure** — no network, no file I/O — except `query.py`, which reads room artifacts, and `cli.py`. This mirrors how `projection.py` and `fusion.py` are already structured and is what makes a geometry bug distinguishable from a VLM bug.
- **TDD is mandatory.** Write the failing test, run it, watch it fail for the right reason, then implement. A test that passes on first run is a plan failure — fix the test.
- **Run tests with** `.venv/Scripts/python.exe -m pytest` (Windows; the venv is not activated in this shell).
- **Coordinate conventions, fixed:** cameras are OpenCV (+X right, +Y **down** the image, +Z forward). `poses[i]` is camera-to-world 4×4. `recon.image_hw` is `(H, W)`. `box_px` is `(x0, y0, x1, y1)` with `x1`/`y1` exclusive. World is Y-up with the floor at y=0 after levelling.
- **Every commit message ends with:**
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  ```
- **Work on branch `object-query`.** It already exists and holds the spec.

### Note on the probe numbers in the spec

The spec quotes measurements (cabinet volume 0.247 → 0.124) taken **without** leave-one-out voting. This plan adds leave-one-out (Task 2): a point proposed by frame 6's box is not allowed to vote for itself in frame 6. That is the honest formulation and it is strictly stricter, so the implemented carving will trim slightly more than the spec's table. Do not treat the spec's numbers as regression targets.

---

### Task 1: `camera.py` — world → pixel, and the visibility test

The repo can currently only go pixel → world. Everything downstream needs the inverse, plus the ability to say whether a point was actually *seen* in a frame or was hidden behind something.

**Files:**
- Create: `src/room3d/camera.py`
- Test: `tests/test_camera.py`

**Interfaces:**
- Consumes: `room3d.artifacts.Reconstruction` (fields `poses`, `intrinsics`, `pts3d`, `conf_mask`, property `image_hw`).
- Produces:
  - `project_to_frame(points, pose_c2w, K, hw) -> tuple[np.ndarray, np.ndarray]` — `(uv, in_view)`, `uv` shape `(M, 2)` as `(u, v)` = `(x, y)` pixels, `in_view` shape `(M,)` bool.
  - `visible_in_frame(points, recon, frame, *, tol=0.10) -> tuple[np.ndarray, np.ndarray]` — `(uv, visible)`. Returns `uv` as well because every caller needs both and projecting twice is waste.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_camera.py`:

```python
"""Gate 6: world -> pixel, and whether a point was actually seen.

Everything else in this repo goes pixel -> world. These are the inverse, and
the visibility test, which is the difference between "that frame disagrees"
and "that frame could not see it". Conflating those destroys small objects.
"""

import numpy as np
import pytest

from room3d.artifacts import Reconstruction
from room3d.camera import project_to_frame, visible_in_frame

H, W = 32, 48
K = np.array([[40.0, 0.0, W / 2], [0.0, 40.0, H / 2], [0.0, 0.0, 1.0]])


def identity_pose():
    """Camera at the origin looking down +Z, OpenCV convention."""
    return np.eye(4)


def make_recon(pts3d, conf=None, pose=None):
    """A one-frame Reconstruction carrying the given pointmap."""
    pts3d = np.asarray(pts3d, dtype=np.float32).reshape(1, H, W, 3)
    return Reconstruction(
        images=np.zeros((1, H, W, 3), dtype=np.uint8),
        pts3d=pts3d,
        conf_mask=np.ones((1, H, W), dtype=bool) if conf is None else conf.reshape(1, H, W),
        poses=np.asarray(pose if pose is not None else identity_pose(),
                         dtype=np.float32).reshape(1, 4, 4),
        intrinsics=np.asarray(K, dtype=np.float32).reshape(1, 3, 3),
        frame_ids=np.zeros(1, dtype=np.int32),
    )


def flat_wall(depth):
    """A pointmap where every pixel's 3D point sits on a plane at z = depth."""
    ys, xs = np.mgrid[0:H, 0:W]
    x = (xs - K[0, 2]) * depth / K[0, 0]
    y = (ys - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, np.full_like(x, depth, dtype=float)], axis=-1)


# --- projection ---------------------------------------------------------------


def test_projecting_a_pointmap_back_into_its_own_frame_is_the_identity():
    """The whole approach rests on this. If it does not hold, nothing downstream
    means anything."""
    wall = flat_wall(2.0).reshape(-1, 3)
    uv, in_view = project_to_frame(wall, identity_pose(), K, (H, W))

    ys, xs = np.mgrid[0:H, 0:W]
    expected = np.stack([xs.ravel(), ys.ravel()], axis=1)
    assert in_view.all()
    assert np.abs(uv - expected).max() < 1e-6


def test_points_behind_the_camera_are_not_in_view():
    uv, in_view = project_to_frame(np.array([[0.0, 0.0, -3.0]]), identity_pose(), K, (H, W))
    assert not in_view[0]


def test_points_outside_the_image_are_not_in_view():
    # 2 m away, far off to the side: lands well beyond the right edge.
    uv, in_view = project_to_frame(np.array([[5.0, 0.0, 2.0]]), identity_pose(), K, (H, W))
    assert uv[0, 0] > W
    assert not in_view[0]


def test_a_point_at_zero_depth_does_not_raise_or_return_nan_as_in_view():
    uv, in_view = project_to_frame(np.array([[0.0, 0.0, 0.0]]), identity_pose(), K, (H, W))
    assert not in_view[0]


def test_projection_respects_a_translated_camera():
    """A camera moved +1 in x sees a point at x=1 dead centre."""
    pose = identity_pose()
    pose[0, 3] = 1.0
    uv, in_view = project_to_frame(np.array([[1.0, 0.0, 2.0]]), pose, K, (H, W))
    assert in_view[0]
    assert uv[0] == pytest.approx([K[0, 2], K[1, 2]], abs=1e-6)


# --- visibility ---------------------------------------------------------------


def test_a_point_on_the_visible_surface_is_visible():
    recon = make_recon(flat_wall(2.0))
    _, seen = visible_in_frame(np.array([[0.0, 0.0, 2.0]]), recon, 0)
    assert seen[0]


def test_a_point_behind_the_visible_surface_is_not_visible():
    """The frame sees a wall at 2 m. A point at 5 m on the same ray is hidden,
    so this frame has no opinion about it."""
    recon = make_recon(flat_wall(2.0))
    _, seen = visible_in_frame(np.array([[0.0, 0.0, 5.0]]), recon, 0)
    assert not seen[0]


def test_a_point_in_front_of_the_visible_surface_is_visible():
    """Nothing occludes it, so the frame can judge it."""
    recon = make_recon(flat_wall(5.0))
    _, seen = visible_in_frame(np.array([[0.0, 0.0, 2.0]]), recon, 0)
    assert seen[0]


def test_the_tolerance_forgives_reconstruction_noise():
    """A point 5% beyond the surface is the same surface, measured imprecisely."""
    recon = make_recon(flat_wall(2.0))
    _, seen = visible_in_frame(np.array([[0.0, 0.0, 2.1]]), recon, 0, tol=0.10)
    assert seen[0]

    _, strict = visible_in_frame(np.array([[0.0, 0.0, 2.1]]), recon, 0, tol=0.01)
    assert not strict[0]


def test_a_point_landing_on_a_low_confidence_pixel_is_not_visible():
    """No usable geometry there means no usable opinion."""
    conf = np.zeros((H, W), dtype=bool)
    recon = make_recon(flat_wall(2.0), conf=conf)
    _, seen = visible_in_frame(np.array([[0.0, 0.0, 2.0]]), recon, 0)
    assert not seen[0]


def test_a_point_outside_the_image_is_not_visible():
    recon = make_recon(flat_wall(2.0))
    _, seen = visible_in_frame(np.array([[5.0, 0.0, 2.0]]), recon, 0)
    assert not seen[0]


def test_visibility_returns_pixel_coordinates_too():
    """Callers always need both; projecting twice would be waste."""
    recon = make_recon(flat_wall(2.0))
    uv, seen = visible_in_frame(np.array([[0.0, 0.0, 2.0]]), recon, 0)
    assert uv.shape == (1, 2)
    assert uv[0] == pytest.approx([K[0, 2], K[1, 2]], abs=1e-6)


def test_empty_input_returns_empty_arrays():
    recon = make_recon(flat_wall(2.0))
    uv, seen = visible_in_frame(np.zeros((0, 3)), recon, 0)
    assert uv.shape == (0, 2)
    assert seen.shape == (0,)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'room3d.camera'`.

- [ ] **Step 3: Write the implementation**

Create `src/room3d/camera.py`:

```python
"""World -> pixel, and whether a point was actually seen.

Every other geometry module here runs pixel -> world: a detection's pixels index
into a pointmap that is already in world coordinates. This module is the inverse,
and it exists because the interesting question about a 3D point is not where it
came from but what *other* cameras have to say about it.

Two functions, and the second is the load-bearing one. Projecting a point into a
frame tells you which pixel it lands on. That is not enough: a point can land
inside a sofa's 2D box while being three metres behind the sofa, hidden by it. A
frame that could not see a point must not get to vote on it, and conflating "this
frame disagrees" with "this frame could not see it" is what destroys small
objects observed by few cameras.

The occlusion test is a z-buffer comparison against the frame's own pointmap:
a point farther from the camera than the surface that frame recorded at that
pixel was behind something. No new data is needed -- the depth buffer is the
reconstruction itself.
"""

from __future__ import annotations

import numpy as np


def project_to_frame(
    points: np.ndarray,
    pose_c2w: np.ndarray,
    K: np.ndarray,
    hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """World points -> `(uv, in_view)` for one camera.

    `uv` is `(M, 2)` in `(x, y)` pixel order. `in_view` marks points that are in
    front of the camera and land inside the image; the `uv` of everything else is
    meaningless and must not be used.

    OpenCV convention throughout: `pose_c2w` is camera-to-world, so the world ->
    camera transform is `R.T @ (x - t)`, and +Y runs *down* the image.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_c2w, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    height, width = int(hw[0]), int(hw[1])

    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    R, t = pose[:3, :3], pose[:3, 3]
    cam = (pts - t[None, :]) @ R            # R.T @ (x - t), vectorised

    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]

    uv = np.stack([u, v], axis=1)
    in_view = (
        (z > 1e-9)
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    return uv, in_view


def visible_in_frame(
    points: np.ndarray,
    recon,
    frame: int,
    *,
    tol: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """`(uv, visible)` -- was each point actually seen by `frame`?

    Visible means: in front of the camera, inside the image, landing on a pixel
    with usable geometry, and no farther from the camera than the surface that
    frame recorded there. `tol` is a *relative* slack on that comparison, so it
    scales with distance -- reconstruction depth error does too, and a fixed
    metric slack would be far too tight nearby and far too loose across a room.

    Returns `uv` alongside the mask because every caller needs the pixel
    coordinates as well, and projecting twice would be pure waste.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    height, width = recon.image_hw

    uv, in_view = project_to_frame(
        pts, recon.poses[frame], recon.intrinsics[frame], (height, width)
    )
    if len(pts) == 0:
        return uv, in_view

    # Clipped so the lookup is always in bounds; `in_view` discards the results
    # for anything that was actually outside.
    ui = np.clip(uv[:, 0], 0, width - 1).astype(np.intp)
    vi = np.clip(uv[:, 1], 0, height - 1).astype(np.intp)

    centre = np.asarray(recon.poses[frame], dtype=np.float64)[:3, 3]
    depth = np.linalg.norm(pts - centre[None, :], axis=1)

    surface = np.asarray(recon.pts3d[frame], dtype=np.float64)[vi, ui]
    surface_depth = np.linalg.norm(surface - centre[None, :], axis=1)
    usable = recon.conf_mask[frame][vi, ui] & np.isfinite(surface_depth)

    return uv, in_view & usable & (depth <= surface_depth * (1.0 + tol))
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 5: Run the whole suite to confirm nothing regressed**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/room3d/camera.py tests/test_camera.py
git commit -m "Add camera.py: world-to-pixel projection and the visibility test

The repo could only go pixel -> world. Cross-view reasoning needs the
inverse, plus a z-buffer test against each frame's own pointmap so a frame
that could not see a point does not get to vote on it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `consensus.py` — the vote

The core primitive. Given some 3D points and a set of `(frame, box)` views, count for each point how many frames that could see it agree that it is inside the box.

**Files:**
- Create: `src/room3d/consensus.py`
- Test: `tests/test_consensus.py`

**Interfaces:**
- Consumes: `room3d.camera.visible_in_frame`.
- Produces:
  - `View` — frozen dataclass: `frame_idx: int`, `box_px: tuple[int, int, int, int]`, `label: str = ""`, `vlm_confidence: float = 1.0`, `observation_id: int | None = None`, `object_id: str | None = None`.
  - `vote(points, views, recon, *, occlusion_tol=0.10, source=None) -> tuple[np.ndarray, np.ndarray]` — `(agree, testable)`, both float arrays of shape `(M,)`.
  - `consistent(points, views, recon, *, min_vote=0.6, occlusion_tol=0.10, source=None) -> np.ndarray` — bool mask of shape `(M,)`.

**Leave-one-out.** `source` is an optional `(M,)` int array giving the frame each point was harvested from. When supplied, a point gets no vote from its own source frame. Without this a point trivially agrees with the box it came from, which inflates every score by exactly one frame and makes a two-view object look half-verified when nothing has verified it at all.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_consensus.py`:

```python
"""Gate 7: cross-view agreement.

A 2D box is axis-aligned and the object inside it is not, so the box contains
things that are not the object. The fix is other cameras: clutter at a different
depth from the object projects outside the box from a different angle. These
tests pin both what that catches and what it cannot.
"""

import numpy as np
import pytest

from room3d.artifacts import Reconstruction
from room3d.consensus import View, consistent, vote

H, W = 40, 40
K = np.array([[40.0, 0.0, W / 2], [0.0, 40.0, H / 2], [0.0, 0.0, 1.0]])


def camera_at(x, z=0.0):
    """A camera on the x/z plane at height 0, looking down +Z."""
    pose = np.eye(4)
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def make_recon(poses, depth=100.0):
    """Cameras with an effectively infinite backdrop, so nothing occludes.

    Occlusion is tested in test_camera.py; these tests are about agreement.
    """
    n = len(poses)
    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.zeros((n, H, W, 3), dtype=np.float32)
    for i, pose in enumerate(poses):
        x = (xs - K[0, 2]) * depth / K[0, 0] + pose[0, 3]
        y = (ys - K[1, 2]) * depth / K[1, 1]
        pts[i] = np.stack([x, y, np.full_like(x, depth + pose[2, 3])], axis=-1)
    return Reconstruction(
        images=np.zeros((n, H, W, 3), dtype=np.uint8),
        pts3d=pts,
        conf_mask=np.ones((n, H, W), dtype=bool),
        poses=np.asarray(poses, dtype=np.float32).reshape(n, 4, 4),
        intrinsics=np.tile(np.asarray(K, dtype=np.float32), (n, 1, 1)),
        frame_ids=np.arange(n, dtype=np.int32),
    )


def box_around(point, recon, frame, half=6):
    """The pixel box centred on where `point` lands in `frame`."""
    from room3d.camera import project_to_frame

    uv, _ = project_to_frame(np.asarray([point], float), recon.poses[frame],
                             recon.intrinsics[frame], recon.image_hw)
    u, v = uv[0]
    return (int(u - half), int(v - half), int(u + half), int(v + half))


# --- the core behaviour --------------------------------------------------------


def test_a_point_inside_every_box_gets_a_full_vote():
    recon = make_recon([camera_at(-1.0), camera_at(0.0), camera_at(1.0)])
    target = np.array([[0.0, 0.0, 4.0]])
    views = [View(i, box_around(target[0], recon, i)) for i in range(3)]

    agree, testable = vote(target, views, recon)
    assert testable[0] == 3
    assert agree[0] == 3


def test_clutter_at_a_different_depth_is_voted_out_by_the_other_views():
    """The whole idea, in one test. A distractor sits inside the box in frame 1
    but parallax carries it outside the box in frames 0 and 2."""
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    distractor = np.array([[0.0, 0.0, 1.5]])       # same ray from frame 1, nearer
    agree, testable = vote(distractor, views, recon)

    assert testable[0] == 3
    assert agree[0] == 1                            # only the frame it hides in
    assert not consistent(distractor, views, recon, min_vote=0.6)[0]
    assert consistent(target[None, :], views, recon, min_vote=0.6)[0]


def test_clutter_inside_the_box_in_every_view_survives():
    """The documented limit. With one camera there is no parallax and nothing
    can be carved. Discovering this in the field would be much worse than
    reading it here."""
    recon = make_recon([camera_at(0.0)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(0, box_around(target, recon, 0))]

    distractor = np.array([[0.0, 0.0, 1.5]])
    assert consistent(distractor, views, recon, min_vote=0.6)[0]


def test_a_point_no_frame_can_see_is_never_consistent():
    recon = make_recon([camera_at(0.0), camera_at(1.0)])
    views = [View(i, (0, 0, W, H)) for i in range(2)]

    behind = np.array([[0.0, 0.0, -5.0]])
    agree, testable = vote(behind, views, recon)
    assert testable[0] == 0
    assert not consistent(behind, views, recon)[0]


# --- min_vote ------------------------------------------------------------------


def test_min_vote_is_monotonic():
    """Raising the bar can only remove points, never add them."""
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    rng = np.random.default_rng(0)
    cloud = target + rng.normal(0, 0.4, (400, 3))

    counts = [
        int(consistent(cloud, views, recon, min_vote=m).sum())
        for m in (0.0, 0.34, 0.67, 1.0)
    ]
    assert counts == sorted(counts, reverse=True)


def test_unanimity_is_available_but_is_not_the_default():
    """It was the default in the first probe and it destroyed a speaker seen by
    two cameras."""
    import inspect

    assert inspect.signature(consistent).parameters["min_vote"].default == 0.6


# --- leave-one-out -------------------------------------------------------------


def test_a_point_does_not_vote_for_itself_when_its_source_frame_is_known():
    """A point harvested from frame 1's box is trivially inside frame 1's box.
    Counting that as verification inflates every score by one frame."""
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    distractor = np.array([[0.0, 0.0, 1.5]])
    source = np.array([1])                          # it came from frame 1

    agree, testable = vote(distractor, views, recon, source=source)
    assert testable[0] == 2                         # frame 1 abstains
    assert agree[0] == 0                            # and it had been the only yes


def test_without_a_source_every_frame_votes():
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    _, testable = vote(np.array([[0.0, 0.0, 4.0]]), views, recon)
    assert testable[0] == 3


# --- shapes and degenerate input -----------------------------------------------


def test_empty_points_give_empty_results():
    recon = make_recon([camera_at(0.0)])
    agree, testable = vote(np.zeros((0, 3)), [View(0, (0, 0, W, H))], recon)
    assert agree.shape == (0,)
    assert testable.shape == (0,)


def test_no_views_means_nothing_is_testable():
    recon = make_recon([camera_at(0.0)])
    agree, testable = vote(np.array([[0.0, 0.0, 4.0]]), [], recon)
    assert testable[0] == 0
    assert not consistent(np.array([[0.0, 0.0, 4.0]]), [], recon)[0]


def test_view_carries_the_detection_provenance_it_came_from():
    v = View(3, (1, 2, 3, 4), label="couch", vlm_confidence=0.8,
             observation_id=7, object_id="obj_002")
    assert (v.frame_idx, v.box_px, v.label) == (3, (1, 2, 3, 4), "couch")
    assert (v.vlm_confidence, v.observation_id, v.object_id) == (0.8, 7, "obj_002")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'room3d.consensus'`.

- [ ] **Step 3: Write the implementation**

Create `src/room3d/consensus.py`:

```python
"""Do the other cameras agree that this point is the object?

A 2D detection box is axis-aligned; the object inside it is not, and is not
rectangular either. So the box always contains things that are not the object --
the sofa standing in front of the cabinet, a strip of floor, the wall behind. No
amount of work on the *box* fixes this, because the information is not in that
frame.

It is in the other frames. Clutter sits at a different depth from the object, so
under parallax it projects outside the box seen from another angle, while the
object projects inside every one of them. Counting those agreements is the whole
method, and it needs nothing that is not already on disk.

Two rules make the count mean something:

- A frame that could not *see* a point does not vote on it. Occlusion is not
  disagreement (see `camera.visible_in_frame`).
- A point does not vote for itself. A point harvested from frame 6's box is
  trivially inside frame 6's box; counting that as verification inflates every
  score by exactly one frame, so a two-view object looks half-confirmed when
  nothing has confirmed it.

`min_vote` is a *fraction* of the frames that could see the point, never a veto.
Requiring unanimity was the first thing tried and it destroyed a speaker seen by
two cameras.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .camera import visible_in_frame


@dataclass(frozen=True)
class View:
    """One 2D box in one frame -- the cached artifact this all rests on.

    The provenance fields are carried, not used, by the geometry here. They are
    what lets a query say which stored observations and which existing objects a
    result subsumes.
    """

    frame_idx: int
    box_px: tuple[int, int, int, int]        # x0, y0, x1, y1; x1/y1 exclusive
    label: str = ""
    vlm_confidence: float = 1.0
    observation_id: int | None = None
    object_id: str | None = None


def vote(
    points: np.ndarray,
    views: Sequence[View],
    recon,
    *,
    occlusion_tol: float = 0.10,
    source: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per point: how many views agree, and how many were entitled to an opinion.

    Returns `(agree, testable)`. Both counts matter -- a point with 2 of 2 is
    better evidenced than one with 2 of 12, and dividing too early throws that
    away.

    `source[i]` is the frame point `i` was harvested from; that frame abstains.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    agree = np.zeros(len(pts))
    testable = np.zeros(len(pts))
    if len(pts) == 0:
        return agree, testable

    src = None if source is None else np.asarray(source).reshape(-1)

    for view in views:
        uv, seen = visible_in_frame(pts, recon, view.frame_idx, tol=occlusion_tol)
        if src is not None:
            seen = seen & (src != view.frame_idx)

        x0, y0, x1, y1 = view.box_px
        inside = (
            seen
            & (uv[:, 0] >= x0) & (uv[:, 0] < x1)
            & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
        )
        agree += inside
        testable += seen

    return agree, testable


def consistent(
    points: np.ndarray,
    views: Sequence[View],
    recon,
    *,
    min_vote: float = 0.6,
    occlusion_tol: float = 0.10,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """Which points enough of the entitled views agree on.

    A point no view could judge is never consistent. Absence of evidence is not
    evidence, and treating it as agreement would keep exactly the points that
    nothing has checked.
    """
    agree, testable = vote(
        points, views, recon, occlusion_tol=occlusion_tol, source=source
    )
    judged = testable > 0
    fraction = np.divide(agree, np.maximum(testable, 1.0))
    return judged & (fraction >= min_vote - 1e-9)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 5: Commit**

```bash
git add src/room3d/consensus.py tests/test_consensus.py
git commit -m "Add consensus.py: cross-view voting on 3D points

Clutter inside a 2D box sits at a different depth from the object, so it
projects outside the box from another angle. Counting agreements carves it
away. Frames that could not see a point abstain, and a point does not vote
for itself.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `consensus.py` — harvesting candidates and carving

Turn a set of views into the tight point set for the object they describe.

**Files:**
- Modify: `src/room3d/consensus.py`
- Modify: `tests/test_consensus.py`

**Interfaces:**
- Consumes: `View`, `consistent` from Task 2.
- Produces:
  - `candidate_points(views, recon) -> tuple[np.ndarray, np.ndarray]` — `(points (M,3), source (M,))`.
  - `largest_component(points, *, voxel=0.04, max_cells=20_000_000) -> np.ndarray` — bool mask `(M,)`.
  - `CarveResult` dataclass: `points: np.ndarray (K,3)`, `source: np.ndarray (K,)`, `stats: dict`.
  - `carve(views, recon, *, min_vote=0.6, occlusion_tol=0.10, keep_largest=True, voxel=0.04) -> CarveResult`.

**Why `carve` returns the sources and not just the points.** Callers need to report how strongly the surviving points were agreed on, and that number is only honest with leave-one-out — which needs to know which frame each surviving point came from. Returning bare points would force every caller either to recompute the carve or to quote a self-inflated vote.

**One-view behaviour:** with a single view, leave-one-out leaves nothing testable and carving would return zero points. `carve` therefore returns the raw candidates unchanged when `len(views) < 2`. That is honest — nothing has been cross-checked — and callers surface the view count so the weakness is visible.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_consensus.py`:

```python
# --- harvesting and carving ----------------------------------------------------

from room3d.consensus import candidate_points, carve, largest_component


def test_candidate_points_gathers_the_pixels_inside_every_box():
    recon = make_recon([camera_at(0.0), camera_at(1.0)])
    views = [View(0, (10, 10, 20, 20)), View(1, (5, 5, 10, 10))]

    points, source = candidate_points(views, recon)
    assert len(points) == 10 * 10 + 5 * 5
    assert (source[: 10 * 10] == 0).all()
    assert (source[10 * 10 :] == 1).all()


def test_candidate_points_skips_low_confidence_pixels():
    recon = make_recon([camera_at(0.0)])
    recon.conf_mask[0, 10:15, 10:20] = False
    points, _ = candidate_points([View(0, (10, 10, 20, 20))], recon)
    assert len(points) == 10 * 10 - 5 * 10


def test_candidate_points_with_no_views_is_empty_but_well_shaped():
    recon = make_recon([camera_at(0.0)])
    points, source = candidate_points([], recon)
    assert points.shape == (0, 3)
    assert source.shape == (0,)


def test_carving_removes_the_distractor_and_keeps_the_object():
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    raw, _ = candidate_points(views, recon)
    result = carve(views, recon, keep_largest=False)
    assert 0 < len(result.points) < len(raw)


def test_carving_reports_where_each_surviving_point_came_from():
    """Callers need this to quote an honest vote fraction, which requires
    leave-one-out, which requires knowing each point's source frame."""
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    result = carve(views, recon, keep_largest=False)
    assert result.source.shape == (len(result.points),)
    assert set(np.unique(result.source)) <= {0, 1, 2}


def test_carving_reports_how_much_it_removed():
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    stats = carve(views, recon, keep_largest=False).stats
    assert stats["n_kept"] < stats["n_candidates"]
    assert 0.0 <= stats["kept_frac"] <= 1.0
    assert 0.0 <= stats["mean_vote"] <= 1.0
    assert stats["min_vote"] == 0.6


def test_carving_a_single_view_returns_its_candidates_unchanged():
    """Leave-one-out leaves nothing to check against. Returning an empty result
    would read as 'not there' when the truth is 'not verifiable'."""
    recon = make_recon([camera_at(0.0)])
    views = [View(0, (10, 10, 20, 20))]

    raw, _ = candidate_points(views, recon)
    result = carve(views, recon)
    assert len(result.points) == len(raw)
    assert result.stats["mean_vote"] == 0.0        # nothing verified it


def test_largest_component_keeps_the_bigger_of_two_separated_blobs():
    rng = np.random.default_rng(0)
    big = rng.normal(0, 0.05, (500, 3))
    small = rng.normal(0, 0.05, (50, 3)) + np.array([3.0, 0.0, 0.0])
    points = np.vstack([big, small])

    keep = largest_component(points, voxel=0.05)
    assert keep[:500].all()
    assert not keep[500:].any()


def test_largest_component_keeps_everything_when_it_is_one_blob():
    rng = np.random.default_rng(0)
    points = rng.normal(0, 0.05, (300, 3))
    assert largest_component(points, voxel=0.05).all()


def test_largest_component_coarsens_rather_than_allocating_a_vast_grid():
    """A single far outlier must not turn a 4 cm grid into gigabytes."""
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 0.05, (200, 3)), [[500.0, 500.0, 500.0]]])
    keep = largest_component(points, voxel=0.001, max_cells=100_000)
    assert keep.shape == (201,)


def test_largest_component_handles_too_few_points_to_cluster():
    assert largest_component(np.zeros((1, 3))).all()
    assert largest_component(np.zeros((0, 3))).shape == (0,)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus.py -q`
Expected: `ImportError: cannot import name 'candidate_points' from 'room3d.consensus'`.

- [ ] **Step 3: Write the implementation**

Append to `src/room3d/consensus.py`:

```python
def candidate_points(
    views: Sequence[View], recon
) -> tuple[np.ndarray, np.ndarray]:
    """Every confident 3D point inside any of the boxes, plus where each came from.

    The union, not the intersection: a partial view contributes the part of the
    object it can see, and the vote below is what removes what it should not
    have contributed.
    """
    chunks: list[np.ndarray] = []
    sources: list[np.ndarray] = []

    for view in views:
        x0, y0, x1, y1 = view.box_px
        mask = np.zeros(recon.image_hw, dtype=bool)
        mask[y0:y1, x0:x1] = True
        mask &= recon.conf_mask[view.frame_idx]

        pts = np.asarray(recon.pts3d[view.frame_idx], dtype=np.float64)[mask]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts):
            chunks.append(pts)
            sources.append(np.full(len(pts), view.frame_idx, dtype=np.int64))

    if not chunks:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.vstack(chunks), np.concatenate(sources)


def largest_component(
    points: np.ndarray, *, voxel: float = 0.04, max_cells: int = 20_000_000
) -> np.ndarray:
    """Keep only the biggest spatially connected blob.

    Carving removes clutter the other views disagree about. What it cannot remove
    is clutter they all agree about -- a chair standing in front of a cabinet in
    every frame. Often that clutter is nonetheless *spatially separate* from the
    object, and then this finds it: voxelise, label 26-connected components, keep
    the largest.

    It does nothing when the clutter physically touches the object, because then
    they genuinely are one connected mass of points. That limit is real and is
    documented in the spec rather than papered over.

    `max_cells` guards the grid: one far-flung outlier with a fine `voxel` would
    otherwise allocate gigabytes, so the resolution is coarsened until the grid
    fits.
    """
    from scipy import ndimage

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return np.ones(len(pts), dtype=bool)

    span = pts.max(axis=0) - pts.min(axis=0)
    voxel = float(max(voxel, 1e-9))
    while float(np.prod(np.floor(span / voxel) + 3.0)) > max_cells:
        voxel *= 2.0

    idx = np.floor((pts - pts.min(axis=0)) / voxel).astype(np.int64) + 1
    grid = np.zeros(tuple(idx.max(axis=0) + 2), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=int))
    if n <= 1:
        return np.ones(len(pts), dtype=bool)

    per_point = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    counts = np.bincount(per_point, minlength=n + 1)
    counts[0] = 0
    return per_point == int(np.argmax(counts))


@dataclass
class CarveResult:
    """What survived, where each surviving point came from, and how it went.

    `source` is carried out of the carve because the only honest way to report
    how strongly a point set is agreed on is with leave-one-out, and that needs
    each point's originating frame. Without it every caller would have to either
    recompute the carve or quote a vote fraction inflated by self-agreement.
    """

    points: np.ndarray                  # (K, 3)
    source: np.ndarray                  # (K,) originating frame per point
    stats: dict


def carve(
    views: Sequence[View],
    recon,
    *,
    min_vote: float = 0.6,
    occlusion_tol: float = 0.10,
    keep_largest: bool = True,
    voxel: float = 0.04,
) -> CarveResult:
    """The points that are actually the object these views describe.

    With fewer than two views there is nothing to cross-check against, so the
    candidates are returned untouched and `mean_vote` is 0.0 -- nothing verified
    them. Returning an empty set there would read as "the object is not present"
    when the truth is "nothing could verify it"; callers report the view count so
    the difference stays visible.
    """
    points, source = candidate_points(views, recon)
    stats = {
        "min_vote": min_vote,
        "n_candidates": int(len(points)),
        "n_kept": int(len(points)),
        "kept_frac": 1.0,
        "mean_vote": 0.0,
    }

    if len(points) == 0 or len(views) < 2:
        return CarveResult(points, source, stats)

    keep = consistent(
        points, views, recon,
        min_vote=min_vote, occlusion_tol=occlusion_tol, source=source,
    )
    kept, kept_source = points[keep], source[keep]

    if keep_largest and len(kept) >= 2:
        biggest = largest_component(kept, voxel=voxel)
        kept, kept_source = kept[biggest], kept_source[biggest]

    stats["n_kept"] = int(len(kept))
    stats["kept_frac"] = round(float(len(kept) / max(len(points), 1)), 4)
    stats["mean_vote"] = round(_mean_vote(kept, kept_source, views, recon, occlusion_tol), 4)
    return CarveResult(kept, kept_source, stats)


def _mean_vote(points, source, views, recon, occlusion_tol: float) -> float:
    """Average agreement across the points that survived, leave-one-out."""
    if len(points) == 0:
        return 0.0
    agree, testable = vote(
        points, views, recon, occlusion_tol=occlusion_tol, source=source
    )
    judged = testable > 0
    if not judged.any():
        return 0.0
    return float((agree[judged] / testable[judged]).mean())
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 5: Commit**

```bash
git add src/room3d/consensus.py tests/test_consensus.py
git commit -m "Add candidate harvesting, connected components and carve()

carve() turns a set of 2D boxes into the points that are actually the object.
largest_component drops spatially separate blobs the vote could not reach; it
cannot help when clutter physically touches the object, which is tested.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `consensus.py` — the identity test

The same primitive, asked a different question: are these two sets of boxes the same physical object? This is what dissolves the duplicate entries.

**Files:**
- Modify: `src/room3d/consensus.py`
- Modify: `tests/test_consensus.py`

**Interfaces:**
- Consumes: `candidate_points`, `vote` from Tasks 2–3, `room3d.projection.subsample`.
- Produces: `agreement_between(views_a, views_b, recon, *, occlusion_tol=0.10, max_points=2000) -> float` — in `[0, 1]`.

**Symmetric by minimum, not mean.** A small object sitting inside a large one scores near 1.0 in one direction (every speaker point is inside the cabinet's boxes) and low in the other. Averaging would merge them; taking the minimum will not.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_consensus.py`:

```python
# --- the identity test ---------------------------------------------------------

from room3d.consensus import agreement_between


def test_two_views_of_the_same_object_agree_strongly():
    recon = make_recon([camera_at(-1.5), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])

    a = [View(0, box_around(target, recon, 0))]
    b = [View(1, box_around(target, recon, 1))]
    assert agreement_between(a, b, recon) > 0.7


def test_views_of_two_different_objects_do_not_agree():
    recon = make_recon([camera_at(-1.5), camera_at(1.5)])
    left = np.array([-1.0, 0.0, 4.0])
    right = np.array([1.5, 0.0, 4.0])

    a = [View(0, box_around(left, recon, 0))]
    b = [View(1, box_around(right, recon, 1))]
    assert agreement_between(a, b, recon) < 0.3


def test_agreement_is_symmetric():
    recon = make_recon([camera_at(-1.5), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    a = [View(0, box_around(target, recon, 0))]
    b = [View(1, box_around(target, recon, 1))]

    assert agreement_between(a, b, recon) == pytest.approx(
        agreement_between(b, a, recon)
    )


def test_a_small_object_inside_a_large_one_is_not_merged_with_it():
    """Every speaker point is inside the cabinet's box, so the speaker->cabinet
    direction scores high. The reverse does not, and the minimum is what stops
    the two becoming one object."""
    recon = make_recon([camera_at(-1.5), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])

    large = [View(0, box_around(target, recon, 0, half=14))]
    small = [View(1, box_around(target, recon, 1, half=2))]
    assert agreement_between(small, large, recon) < 0.6


def test_two_boxes_in_the_same_frame_never_merge():
    """The detector already said these are two objects. Leave-one-out makes that
    conclusion fall out rather than needing a special case."""
    recon = make_recon([camera_at(0.0)])
    a = [View(0, (5, 5, 15, 15))]
    b = [View(0, (6, 6, 16, 16))]
    assert agreement_between(a, b, recon) == 0.0


def test_agreement_with_an_empty_side_is_zero():
    recon = make_recon([camera_at(0.0)])
    assert agreement_between([], [View(0, (5, 5, 15, 15))], recon) == 0.0
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus.py -q`
Expected: `ImportError: cannot import name 'agreement_between'`.

- [ ] **Step 3: Write the implementation**

Append to `src/room3d/consensus.py`:

```python
def agreement_between(
    views_a: Sequence[View],
    views_b: Sequence[View],
    recon,
    *,
    occlusion_tol: float = 0.10,
    max_points: int = 2000,
) -> float:
    """Do these two sets of boxes describe the same physical object? 0 to 1.

    The same vote as `carve`, asked across two candidates instead of within one.
    If A's points land inside B's boxes wherever B can see them, and B's inside
    A's, they are one object seen twice -- which is exactly the evidence that
    `fusion.py`'s centroid-distance test is only a proxy for.

    Symmetric by taking the **minimum** of the two directions, never the mean. A
    speaker sitting inside a cabinet scores near 1.0 speaker-to-cabinet, because
    every speaker point really is inside the cabinet's box; the mean would merge
    them into one object and the minimum will not.

    Subsampled because grouping compares every pair of candidates, and the
    quadratic term is what would make a query feel slow.
    """
    from .projection import subsample

    def directional(source_views, target_views) -> float:
        points, _ = candidate_points(source_views, recon)
        if len(points) == 0 or not target_views:
            return 0.0
        points = subsample(points, max_points)

        # The source frames abstain: a point is inside the box it came from by
        # construction, and two detections in the same frame are two objects.
        source = np.full(len(points), -1, dtype=np.int64)
        agree, testable = vote(
            points, target_views, recon,
            occlusion_tol=occlusion_tol, source=source,
        )
        judged = testable > 0
        if not judged.any():
            return 0.0
        return float((agree[judged] / testable[judged]).mean())

    frames_a = {v.frame_idx for v in views_a}
    shared = [v for v in views_b if v.frame_idx in frames_a]
    if shared and len(shared) == len(views_b):
        # Every box on one side is in a frame the other side already claims.
        # The detector called those separate objects; nothing here overrules it.
        return 0.0

    b_minus_a = [v for v in views_b if v.frame_idx not in frames_a]
    frames_b = {v.frame_idx for v in views_b}
    a_minus_b = [v for v in views_a if v.frame_idx not in frames_b]

    return min(
        directional(views_a, b_minus_a),
        directional(views_b, a_minus_b),
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/room3d/consensus.py tests/test_consensus.py
git commit -m "Add agreement_between: the identity test

Same vote, asked across two candidates. Symmetric by minimum so a small
object inside a large one is not swallowed by it, and two boxes in one frame
never merge because the detector already called them separate.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `consensus.py` — the per-frame consistency mask

The same computation transposed: score pixels rather than pooled points. This is the geometry-derived mask, and it costs nothing extra.

**Files:**
- Modify: `src/room3d/consensus.py`
- Modify: `tests/test_consensus.py`

**Interfaces:**
- Consumes: `View`, `consistent` from Task 2.
- Produces: `consistency_mask(frame, views, recon, *, min_vote=0.6, occlusion_tol=0.10) -> np.ndarray` — bool `(H, W)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_consensus.py`:

```python
# --- the free mask -------------------------------------------------------------

from room3d.consensus import consistency_mask


def test_the_mask_is_a_subset_of_the_box_it_explains():
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    mask = consistency_mask(1, views, recon)
    assert mask.shape == recon.image_hw

    x0, y0, x1, y1 = views[1].box_px
    outside = mask.copy()
    outside[y0:y1, x0:x1] = False
    assert not outside.any()


def test_the_mask_marks_fewer_pixels_than_the_box_contains():
    """If it marked all of them it would be telling you nothing."""
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i, half=12)) for i in range(3)]

    x0, y0, x1, y1 = views[1].box_px
    assert consistency_mask(1, views, recon).sum() < (x1 - x0) * (y1 - y0)


def test_a_frame_with_no_box_gets_an_empty_mask():
    recon = make_recon([camera_at(0.0), camera_at(1.0)])
    mask = consistency_mask(1, [View(0, (5, 5, 15, 15))], recon)
    assert not mask.any()


def test_raising_min_vote_can_only_shrink_the_mask():
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i, half=12)) for i in range(3)]

    loose = consistency_mask(1, views, recon, min_vote=0.0).sum()
    tight = consistency_mask(1, views, recon, min_vote=1.0).sum()
    assert tight <= loose
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus.py -q`
Expected: `ImportError: cannot import name 'consistency_mask'`.

- [ ] **Step 3: Write the implementation**

Append to `src/room3d/consensus.py`:

```python
def consistency_mask(
    frame: int,
    views: Sequence[View],
    recon,
    *,
    min_vote: float = 0.6,
    occlusion_tol: float = 0.10,
) -> np.ndarray:
    """Which pixels of `frame`'s box the other views agree are the object.

    The same computation as `carve`, transposed: score pixels instead of pooled
    points. That makes it a segmentation mask derived from geometry alone -- no
    model, no extra call -- and it is the only way to *see* whether a box that
    also contains a sofa is being counted as cabinet or as sofa.

    Returns a full-image mask so it can be composited directly onto the frame.
    """
    height, width = recon.image_hw
    region = np.zeros((height, width), dtype=bool)
    for view in views:
        if view.frame_idx == frame:
            x0, y0, x1, y1 = view.box_px
            region[y0:y1, x0:x1] = True

    out = np.zeros((height, width), dtype=bool)
    region &= recon.conf_mask[frame]
    if not region.any():
        return out

    where = np.argwhere(region)                     # (M, 2), rows of (y, x)
    points = np.asarray(recon.pts3d[frame], dtype=np.float64)[region]

    finite = np.isfinite(points).all(axis=1)
    where, points = where[finite], points[finite]
    if len(points) == 0:
        return out

    # These pixels came from `frame`, so `frame` abstains -- otherwise every
    # pixel would agree with itself and the mask would just redraw the box.
    source = np.full(len(points), frame, dtype=np.int64)
    keep = consistent(
        points, views, recon,
        min_vote=min_vote, occlusion_tol=occlusion_tol, source=source,
    )
    out[where[keep, 0], where[keep, 1]] = True
    return out
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 5: Commit**

```bash
git add src/room3d/consensus.py tests/test_consensus.py
git commit -m "Add consistency_mask: the same vote, scored per pixel

A geometry-derived segmentation mask with no model and no extra call. It is
what makes it possible to see whether a cabinet box is being counted as
cabinet or as the sofa standing in front of it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `QueryConfig` and phrase → cached views

The first half of `query.py`: turn a phrase into the views already on disk.

**Files:**
- Create: `src/room3d/query.py`
- Modify: `src/room3d/config.py`
- Test: `tests/test_query_resolve.py`

**Interfaces:**
- Consumes: `room3d.consensus.View`, `room3d.fusion.default_label_compatible`, `room3d.fusion.normalize_label`.
- Produces:
  - `QueryConfig` dataclass in `config.py` with fields `min_vote: float = 0.6`, `occlusion_tol: float = 0.10`, `keep_largest_component: bool = True`, `component_voxel: float = 0.04`, `min_instance_agreement: float = 0.3`, `max_agreement_points: int = 2000`; added to `Config` as `query: QueryConfig`.
  - `normalize_phrase(phrase) -> str` in `query.py`.
  - `cached_views(observations_doc, phrase, *, label_compatible=default_label_compatible) -> list[View]`.

**The qualifier rule.** `normalize_phrase` strips a leading article only. So "the couch" normalises to "couch" and matches; "the couch by the window" normalises to "couch by the window" and matches nothing, which falls through to the VLM in Task 9. That is the specified behaviour and it needs no phrase parser — labels cannot resolve spatial language, and matching on the head noun would silently answer a different question.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_query_resolve.py`:

```python
"""Gate 8a: turning a phrase into the boxes already on disk.

The expensive part of a run is the 2D detections. A query that can answer from
them costs nothing, so this is the path that must be tried first.
"""

import pytest

from room3d.config import Config, QueryConfig, load_config
from room3d.query import cached_views, normalize_phrase


def doc(*entries):
    return {
        "observations": [
            {
                "id": i,
                "object_id": obj,
                "frame_idx": frame,
                "label": label,
                "vlm_confidence": conf,
                "box_px": [1, 2, 3, 4],
            }
            for i, (obj, frame, label, conf) in enumerate(entries)
        ]
    }


SAMPLE = doc(
    ("obj_000", 0, "couch", 0.9),
    ("obj_000", 1, "sofa", 0.8),
    ("obj_001", 2, "chair", 0.7),
    ("obj_002", 3, "chair", 0.6),
)


# --- normalisation -------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("Couch", "couch"),
        ("  the couch  ", "couch"),
        ("A chair", "chair"),
        ("an office chair", "office chair"),
        ("the couch by the window", "couch by the window"),
        ("office_chair", "office chair"),
    ],
)
def test_normalize_phrase(phrase, expected):
    assert normalize_phrase(phrase) == expected


def test_only_a_leading_article_is_stripped():
    """'the' inside the phrase is part of a qualifier and must survive, or the
    qualifier silently disappears and the wrong question gets answered."""
    assert normalize_phrase("the sofa near the door") == "sofa near the door"


# --- matching ------------------------------------------------------------------


def test_a_plain_label_matches_its_detections():
    views = cached_views(SAMPLE, "couch")
    assert {v.frame_idx for v in views} == {0, 1}


def test_synonyms_match_through_the_existing_table():
    """couch and sofa are one object; fusion already knows that."""
    assert {v.frame_idx for v in cached_views(SAMPLE, "sofa")} == {0, 1}


def test_a_label_shared_by_several_objects_returns_all_of_them():
    """Splitting them into instances is a later stage's job, not this one's."""
    views = cached_views(SAMPLE, "chair")
    assert {v.frame_idx for v in views} == {2, 3}
    assert {v.object_id for v in views} == {"obj_001", "obj_002"}


def test_a_qualified_phrase_matches_nothing_so_it_can_fall_through_to_the_vlm():
    assert cached_views(SAMPLE, "the couch by the window") == []


def test_an_unknown_label_matches_nothing():
    assert cached_views(SAMPLE, "aquarium") == []


def test_views_carry_the_provenance_a_commit_will_need():
    view = cached_views(SAMPLE, "couch")[0]
    assert view.observation_id == 0
    assert view.object_id == "obj_000"
    assert view.label == "couch"
    assert view.vlm_confidence == pytest.approx(0.9)
    assert view.box_px == (1, 2, 3, 4)


def test_observations_without_a_pixel_box_are_skipped():
    """Nothing to project. Carrying it forward would fake evidence."""
    d = doc(("obj_000", 0, "couch", 0.9))
    d["observations"][0]["box_px"] = None
    assert cached_views(d, "couch") == []


def test_an_injected_matcher_is_respected():
    """The CrewAI layer swaps in an LLM adjudicator, exactly as fusion does."""
    assert len(cached_views(SAMPLE, "wombat", label_compatible=lambda a, b: True)) == 4


# --- config --------------------------------------------------------------------


def test_query_config_has_the_documented_defaults():
    c = QueryConfig()
    assert c.min_vote == 0.6
    assert c.occlusion_tol == 0.10
    assert c.keep_largest_component is True
    assert c.component_voxel == 0.04
    assert c.min_instance_agreement == 0.3


def test_query_config_is_reachable_from_the_top_level_config():
    assert isinstance(Config().query, QueryConfig)


def test_query_config_loads_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("query:\n  min_vote: 0.8\n")
    assert load_config(path).query.min_vote == 0.8


def test_unknown_query_keys_are_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("query:\n  nonsense: 1\n")
    with pytest.raises(ValueError, match="nonsense"):
        load_config(path)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_resolve.py -q`
Expected: `ImportError: cannot import name 'QueryConfig' from 'room3d.config'`.

- [ ] **Step 3: Add `QueryConfig` to `src/room3d/config.py`**

Insert after the `LabelConfig` dataclass:

```python
@dataclass
class QueryConfig:
    """Knobs for `room3d query`.

    `min_vote` is the one that matters and the one that is genuinely uncertain:
    it is the fraction of the frames that could see a point which must agree it
    is inside the box. Unanimity destroys small objects seen by few cameras;
    zero carves nothing. It is tuned against one room, so every query reports its
    vote statistics rather than leaving this a magic number.
    """

    min_vote: float = 0.6
    occlusion_tol: float = 0.10          # relative depth slack in the z-buffer test
    keep_largest_component: bool = True
    component_voxel: float = 0.04        # metres
    min_instance_agreement: float = 0.3  # above this, two view-sets are one object
    max_agreement_points: int = 2000     # grouping is pairwise; this caps the cost
```

Then extend `Config` and `load_config`:

```python
@dataclass
class Config:
    frames: FramesConfig = field(default_factory=FramesConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
```

```python
def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        frames=_build(FramesConfig, raw.get("frames", {})),
        reconstruct=_build(ReconstructConfig, raw.get("reconstruct", {})),
        label=_build(LabelConfig, raw.get("label", {})),
        query=_build(QueryConfig, raw.get("query", {})),
    )
```

- [ ] **Step 4: Create `src/room3d/query.py` with phrase resolution**

```python
"""Name an object, get its box.

The pipeline's job is to label everything; this is for when you want *one*
thing and want it right. It changes what the geometry is anchored to. Fusion
must guess which observations belong together from centroid distance, and when
it guesses wrong a couch appears twice. A query names the object, so identity is
given rather than inferred, and the machinery is free to spend its effort on
where the object actually is.

Nothing here recomputes the reconstruction. The 2D detections are already on
disk and, per the user, they are good; what was missing was a way to combine
them that respects the fact that a box is not an object.

Order of operations:

    phrase -> views -> instances -> carve -> box -> ranked matches

Cached detections are tried first because they are free. The VLM is called only
when the phrase matches nothing, or when explicitly forced.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .consensus import View
from .fusion import default_label_compatible, normalize_label

_ARTICLES = ("the ", "a ", "an ")


def normalize_phrase(phrase: str) -> str:
    """Lowercase, collapse whitespace, and strip one leading article.

    Only a *leading* article. "the couch" is a request for the couch; "the couch
    by the window" is a request that cached labels cannot evaluate, and stripping
    the inner "the" would not make it evaluable -- it would only make the phrase
    look like it had been understood.
    """
    text = normalize_label(phrase)
    for article in _ARTICLES:
        if text.startswith(article):
            return text[len(article):].strip()
    return text


def cached_views(
    observations_doc: dict,
    phrase: str,
    *,
    label_compatible: Callable[[str, str], bool] = default_label_compatible,
) -> list[View]:
    """Every stored detection whose label matches `phrase`.

    Returns detections belonging to several different objects when the phrase is
    ambiguous ("chair"). Separating them is `group_instances`'s job -- doing it
    here would mean guessing which chair was meant before anything has looked at
    the geometry.

    An empty list means the cache cannot answer, which is the signal to fall
    through to the VLM. That is also how a qualified phrase is handled: it
    matches no label, so it becomes a miss rather than a wrong answer.
    """
    target = normalize_phrase(phrase)
    views: list[View] = []

    for i, item in enumerate(observations_doc.get("observations", [])):
        box = item.get("box_px")
        if not box:
            continue
        if not label_compatible(str(item.get("label", "")), target):
            continue
        views.append(
            View(
                frame_idx=int(item["frame_idx"]),
                box_px=tuple(int(v) for v in box),
                label=str(item.get("label", "")),
                vlm_confidence=float(item.get("vlm_confidence", 0.5)),
                observation_id=int(item.get("id", i)),
                object_id=item.get("object_id"),
            )
        )
    return views
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_resolve.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass. `test_pipeline_wiring.py` and any config tests must be unaffected by the new `Config` field.

- [ ] **Step 7: Commit**

```bash
git add src/room3d/query.py src/room3d/config.py tests/test_query_resolve.py
git commit -m "Add QueryConfig and phrase-to-cached-views resolution

A phrase matching a cached label answers for free. A phrase carrying a
qualifier the labels cannot evaluate matches nothing on purpose, so it falls
through to the VLM instead of silently answering a different question.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Instance grouping

"chair" matches eight detections that are four chairs. Split them using the identity test, not a new algorithm.

**Files:**
- Modify: `src/room3d/query.py`
- Test: `tests/test_query_instances.py`

**Interfaces:**
- Consumes: `room3d.consensus.agreement_between`, `View`.
- Produces: `group_instances(views, recon, *, min_agreement=0.3, occlusion_tol=0.10, max_points=2000) -> list[list[View]]`.

**Ordering.** Views are seeded largest-box-first, so well-supported detections start the groups and marginal ones attach to them. This mirrors the rationale already written into `fusion.cluster_observations`, which sorts by point support for the same reason.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_query_instances.py`:

```python
"""Gate 8b: one phrase, several objects.

"chair" matches every chair in the room. Telling them apart uses the same
cross-view vote as carving, because "are these the same object?" is the
question the vote already answers.
"""

import numpy as np

from room3d.artifacts import Reconstruction
from room3d.camera import project_to_frame
from room3d.consensus import View
from room3d.query import group_instances

H, W = 40, 60
K = np.array([[40.0, 0.0, W / 2], [0.0, 40.0, H / 2], [0.0, 0.0, 1.0]])


def camera_at(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def make_recon(xs, depth=100.0):
    ys, px = np.mgrid[0:H, 0:W]
    pts = np.zeros((len(xs), H, W, 3), dtype=np.float32)
    poses = []
    for i, x in enumerate(xs):
        pose = camera_at(x)
        poses.append(pose)
        wx = (px - K[0, 2]) * depth / K[0, 0] + x
        wy = (ys - K[1, 2]) * depth / K[1, 1]
        pts[i] = np.stack([wx, wy, np.full_like(wx, depth)], axis=-1)
    return Reconstruction(
        images=np.zeros((len(xs), H, W, 3), dtype=np.uint8),
        pts3d=pts,
        conf_mask=np.ones((len(xs), H, W), dtype=bool),
        poses=np.asarray(poses, dtype=np.float32),
        intrinsics=np.tile(np.asarray(K, dtype=np.float32), (len(xs), 1, 1)),
        frame_ids=np.arange(len(xs), dtype=np.int32),
    )


def box_around(point, recon, frame, half=5):
    uv, _ = project_to_frame(np.asarray([point], float), recon.poses[frame],
                             recon.intrinsics[frame], recon.image_hw)
    u, v = uv[0]
    return (int(u - half), int(v - half), int(u + half), int(v + half))


def test_repeated_sightings_of_one_object_form_one_instance():
    recon = make_recon([-1.5, 0.0, 1.5])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i), label="chair") for i in range(3)]

    groups = group_instances(views, recon)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_two_separate_objects_form_two_instances():
    recon = make_recon([-1.5, 0.0, 1.5])
    left = np.array([-1.2, 0.0, 4.0])
    right = np.array([1.2, 0.0, 4.0])

    views = [View(i, box_around(left, recon, i), label="chair") for i in range(3)]
    views += [View(i, box_around(right, recon, i), label="chair") for i in range(3)]

    assert len(group_instances(views, recon)) == 2


def test_two_detections_in_the_same_frame_are_always_separate_instances():
    """The detector already decided. Nothing here overrules it."""
    recon = make_recon([0.0])
    views = [
        View(0, (5, 5, 15, 15), label="chair"),
        View(0, (30, 5, 40, 15), label="chair"),
    ]
    assert len(group_instances(views, recon)) == 2


def test_a_lower_threshold_merges_more():
    recon = make_recon([-1.5, 0.0, 1.5])
    left = np.array([-0.6, 0.0, 4.0])
    right = np.array([0.6, 0.0, 4.0])
    views = [View(i, box_around(left, recon, i), label="chair") for i in range(3)]
    views += [View(i, box_around(right, recon, i), label="chair") for i in range(3)]

    strict = len(group_instances(views, recon, min_agreement=0.95))
    loose = len(group_instances(views, recon, min_agreement=0.0))
    assert loose <= strict


def test_no_views_gives_no_instances():
    assert group_instances([], make_recon([0.0])) == []


def test_a_single_view_gives_one_instance():
    recon = make_recon([0.0])
    assert len(group_instances([View(0, (5, 5, 15, 15))], recon)) == 1


def test_instances_are_ordered_by_support():
    """The best-evidenced answer should be match 1, since --commit defaults to it."""
    recon = make_recon([-1.5, 0.0, 1.5])
    left = np.array([-1.2, 0.0, 4.0])
    right = np.array([1.2, 0.0, 4.0])

    views = [View(i, box_around(left, recon, i), label="chair") for i in range(3)]
    views += [View(0, box_around(right, recon, 0), label="chair")]

    sizes = [len(g) for g in group_instances(views, recon)]
    assert sizes == sorted(sizes, reverse=True)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_instances.py -q`
Expected: `ImportError: cannot import name 'group_instances'`.

- [ ] **Step 3: Write the implementation**

Append to `src/room3d/query.py`:

```python
def _box_area(box_px: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box_px
    return max(x1 - x0, 0) * max(y1 - y0, 0)


def group_instances(
    views: Sequence[View],
    recon,
    *,
    min_agreement: float = 0.3,
    occlusion_tol: float = 0.10,
    max_points: int = 2000,
) -> list[list[View]]:
    """Split the matched detections into distinct physical objects.

    "chair" matches every chair in the room. Which detections are the *same*
    chair is decided by `agreement_between` -- the same cross-view vote used to
    carve -- rather than by centroid distance. That matters because centroid
    distance is only a proxy for the question, and it is the proxy that puts one
    couch in the object list twice.

    Greedy, largest box first, so well-supported detections seed the groups and
    marginal ones attach to them. `fusion.cluster_observations` orders itself the
    same way and for the same reason.
    """
    from .consensus import agreement_between

    groups: list[list[View]] = []

    for view in sorted(views, key=lambda v: -_box_area(v.box_px)):
        best: list[View] | None = None
        best_score = min_agreement

        for group in groups:
            score = agreement_between(
                [view], group, recon,
                occlusion_tol=occlusion_tol, max_points=max_points,
            )
            if score >= best_score:
                best, best_score = group, score

        if best is None:
            groups.append([view])
        else:
            best.append(view)

    return sorted(groups, key=lambda g: -len(g))
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_instances.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 5: Commit**

```bash
git add src/room3d/query.py tests/test_query_instances.py
git commit -m "Add group_instances: split matched detections into objects

Uses agreement_between rather than centroid distance. Centroid distance is a
proxy for 'are these the same object?'; the cross-view vote is the question
itself, and it is the proxy that lists one couch twice.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `query_room` — carve, fit, rank

Assemble the pipeline into the entry point, returning ranked `QueryMatch` objects.

**Files:**
- Modify: `src/room3d/query.py`
- Test: `tests/test_query_room.py`

**Interfaces:**
- Consumes: everything above, plus `room3d.projection.fit_gravity_aligned_box`, `snap_to_floor`, `OrientedBox`; `room3d.level.estimate_up`; `room3d.artifacts.load_frames_npz`.
- Produces:
  - `QueryMatch` dataclass: `label: str`, `obb: OrientedBox | None`, `score: float`, `views: list[View]`, `n_points: int`, `vote_stats: dict`, `absorbed_object_ids: list[str]`, `supported: bool`; method `as_dict() -> dict`.
  - `QueryResult` dataclass: `phrase: str`, `matches: list[QueryMatch]`, `source: str` (`"cache"` | `"vlm"` | `"none"`), `notes: list[str]`; method `as_dict() -> dict`.
  - `query_room(room_dir, phrase, *, config=None, config_overrides=None, detector=None, force=False, label_compatible=default_label_compatible, verbose=True) -> QueryResult`. `config_overrides` is a dict applied over `config` via `dataclasses.replace`; it is what lets the CLI pass `--min-vote` without constructing a whole config.

**Unsupported matches are returned, not dropped.** A match found in 2D whose points are entirely carved away is reported with `supported=False` and its view count. "Found it in the frames, could not stand it up in 3D" is a different and more useful answer than silence.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_query_room.py`:

```python
"""Gate 8c: the whole query, end to end, on a room on disk.

Reuses the synthetic levelled room from the box-fit tests so the query is
exercised against the same geometry the pipeline produces.
"""

import json

import numpy as np
import pytest

from room3d.artifacts import save_frames_npz
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from room3d.level import UP_VECTOR
from room3d.query import QueryMatch, query_room
from test_boxfit_wiring import labelled_session


def room_on_disk(tmp_path):
    session = labelled_session()
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)

    save_frames_npz(tmp_path / "frames.npz", session.recon)
    (tmp_path / "observations.json").write_text(
        json.dumps(build_observations_doc(session, "room"))
    )
    (tmp_path / "objects.json").write_text(
        json.dumps({
            "room": "room", "units": "meters", "scale_verified": False,
            "n_frames_labeled": 4,
            "objects": [o.as_dict() for o in session.objects],
        })
    )
    return tmp_path


def test_querying_a_labelled_object_returns_one_match(tmp_path):
    result = query_room(room_on_disk(tmp_path), "sofa", verbose=False)
    assert result.source == "cache"
    assert len(result.matches) == 1
    assert result.matches[0].label == "sofa"


def test_the_match_carries_a_levelled_box(tmp_path):
    match = query_room(room_on_disk(tmp_path), "sofa", verbose=False).matches[0]
    assert match.obb is not None
    assert np.allclose(match.obb.R[:, 1], UP_VECTOR, atol=1e-2)


def test_a_synonym_finds_the_same_object(tmp_path):
    room = room_on_disk(tmp_path)
    a = query_room(room, "sofa", verbose=False).matches[0]
    b = query_room(room, "couch", verbose=False).matches[0]
    assert np.allclose(a.obb.extent, b.obb.extent)


def test_the_match_names_the_objects_it_would_replace(tmp_path):
    match = query_room(room_on_disk(tmp_path), "sofa", verbose=False).matches[0]
    assert match.absorbed_object_ids == ["obj_000"]


def test_an_unknown_phrase_with_no_detector_says_so_rather_than_saying_absent(tmp_path):
    result = query_room(room_on_disk(tmp_path), "aquarium", verbose=False)
    assert result.matches == []
    assert result.source == "none"
    assert any("no detector" in n for n in result.notes)


def test_vote_statistics_are_reported_so_min_vote_is_not_a_magic_number(tmp_path):
    match = query_room(room_on_disk(tmp_path), "sofa", verbose=False).matches[0]
    assert set(match.vote_stats) >= {"min_vote", "n_candidates", "n_kept", "kept_frac"}
    assert 0.0 <= match.vote_stats["kept_frac"] <= 1.0


def test_carving_removes_points_the_views_disagree_about(tmp_path):
    match = query_room(room_on_disk(tmp_path), "sofa", verbose=False).matches[0]
    assert match.vote_stats["n_kept"] < match.vote_stats["n_candidates"]


def test_a_match_whose_points_are_all_carved_away_is_reported_as_unsupported(tmp_path):
    """Found in 2D, could not be stood up in 3D. Silence would be a worse answer."""
    result = query_room(
        room_on_disk(tmp_path), "sofa",
        config_overrides={"min_vote": 1.01}, verbose=False,
    )
    assert len(result.matches) == 1
    assert result.matches[0].supported is False
    assert result.matches[0].obb is None


def test_matches_are_ranked_best_first(tmp_path):
    result = query_room(room_on_disk(tmp_path), "sofa", verbose=False)
    scores = [m.score for m in result.matches]
    assert scores == sorted(scores, reverse=True)


def test_the_result_is_json_serialisable(tmp_path):
    result = query_room(room_on_disk(tmp_path), "sofa", verbose=False)
    json.dumps(result.as_dict())          # must not raise


def test_a_missing_room_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="frames.npz"):
        query_room(tmp_path, "sofa", verbose=False)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_room.py -q`
Expected: `ImportError: cannot import name 'QueryMatch'`.

- [ ] **Step 3: Write the implementation**

Append to `src/room3d/query.py` (and add the imports shown at the top of the block to the module's existing import section):

```python
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .artifacts import load_frames_npz
from .config import QueryConfig
from .consensus import carve
from .level import estimate_up
from .projection import OrientedBox, fit_gravity_aligned_box, snap_to_floor

# Matches LabelConfig.min_level_confidence. Below this the levelling estimate is
# not worth acting on, and a box levelled against a wrong gravity vector is worse
# than an unlevelled one.
MIN_LEVEL_CONFIDENCE = 0.25


@dataclass
class QueryMatch:
    """One physical object the query found, and the evidence behind it."""

    label: str
    obb: OrientedBox | None
    score: float
    views: list[View]
    n_points: int
    vote_stats: dict
    absorbed_object_ids: list[str] = field(default_factory=list)
    supported: bool = True

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "obb": self.obb.as_dict() if self.obb is not None else None,
            "score": round(float(self.score), 4),
            "n_views": len(self.views),
            "n_points": int(self.n_points),
            "frames": sorted({v.frame_idx for v in self.views}),
            "boxes": {str(v.frame_idx): list(v.box_px) for v in self.views},
            "vote_stats": self.vote_stats,
            "absorbed_object_ids": list(self.absorbed_object_ids),
            "supported": bool(self.supported),
        }


@dataclass
class QueryResult:
    phrase: str
    matches: list[QueryMatch]
    source: str                       # "cache" | "vlm" | "none"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "phrase": self.phrase,
            "source": self.source,
            "matches": [m.as_dict() for m in self.matches],
            "notes": list(self.notes),
        }


def query_room(
    room_dir: str | Path,
    phrase: str,
    *,
    config: QueryConfig | None = None,
    config_overrides: dict | None = None,
    detector=None,
    force: bool = False,
    label_compatible: Callable[[str, str], bool] = default_label_compatible,
    verbose: bool = True,
) -> QueryResult:
    """Find `phrase` in a labelled room and return its box, best match first.

    Reads only. Promoting a match into `objects.json` is `commit_match`'s job,
    and keeping them separate is what makes a vague query free rather than
    destructive.
    """
    room_dir = Path(room_dir)
    npz = room_dir / "frames.npz"
    obs_path = room_dir / "observations.json"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found; reconstruct this room first")
    if not obs_path.exists():
        raise FileNotFoundError(f"{obs_path} not found; label this room first")

    config = config or QueryConfig()
    if config_overrides:
        config = replace(config, **config_overrides)

    recon = load_frames_npz(npz)
    doc = json.loads(obs_path.read_text())
    notes: list[str] = []

    views: list[View] = []
    source = "none"
    if not force:
        views = cached_views(doc, phrase, label_compatible=label_compatible)
        if views:
            source = "cache"

    if not views:
        if detector is None:
            notes.append(
                "nothing cached matches that phrase, and no detector is "
                "available to look for it -- this is not the same as the object "
                "being absent from the room"
            )
            return QueryResult(phrase, [], "none", notes)
        views = _vlm_views(recon, doc, phrase, detector, notes)
        source = "vlm"

    up, floor_height = _scene_frame(recon, notes)

    matches = [
        _build_match(group, recon, up, floor_height, config)
        for group in group_instances(
            views, recon,
            min_agreement=config.min_instance_agreement,
            occlusion_tol=config.occlusion_tol,
            max_points=config.max_agreement_points,
        )
    ]
    matches.sort(key=lambda m: -m.score)

    if verbose:
        print(f"[query] {phrase!r} -> {len(matches)} match(es) from {source}")
        for i, m in enumerate(matches, 1):
            size = "unsupported" if m.obb is None else np.round(m.obb.extent, 3).tolist()
            print(f"[query]   {i}. {m.label:<16} score={m.score:.2f} "
                  f"views={len(m.views)} extent={size}")
        for note in notes:
            print(f"[query] note: {note}")

    return QueryResult(phrase, matches, source, notes)


def _scene_frame(recon, notes: list[str]):
    """`(up, floor_height)` for the room, or `(None, None)` if untrustworthy."""
    estimate = estimate_up(recon.pts3d[recon.conf_mask], recon.poses)
    if estimate.confidence < MIN_LEVEL_CONFIDENCE:
        notes.append(
            f"up estimate too weak to trust ({estimate.confidence:.2f}); "
            "the box will not be levelled"
        )
        return None, None
    return estimate.up, float(estimate.floor_offset)


def _build_match(group, recon, up, floor_height, config: QueryConfig) -> QueryMatch:
    """Carve one instance's points and wrap a box around what survives."""
    result = carve(
        group, recon,
        min_vote=config.min_vote,
        occlusion_tol=config.occlusion_tol,
        keep_largest=config.keep_largest_component,
        voxel=config.component_voxel,
    )
    kept, stats = result.points, result.stats

    absorbed = sorted({v.object_id for v in group if v.object_id})
    label = _dominant_label(group)

    if len(kept) < 3:
        # Found in 2D and unsupportable in 3D. Reporting it as a match with
        # supported=False says that; returning nothing would say "not present".
        return QueryMatch(label, None, 0.0, list(group), 0, stats, absorbed, False)

    if up is None:
        from .projection import fit_oriented_box

        obb = fit_oriented_box(kept)
    else:
        obb = fit_gravity_aligned_box(kept, up)
        if floor_height is not None:
            obb = snap_to_floor(obb, floor_height, up)

    return QueryMatch(
        label=label,
        obb=obb,
        score=_score(group, stats["mean_vote"]),
        views=list(group),
        n_points=int(len(kept)),
        vote_stats=stats,
        absorbed_object_ids=absorbed,
        supported=True,
    )


def _dominant_label(group: Sequence[View]) -> str:
    counts: dict[str, int] = {}
    for view in group:
        name = normalize_label(view.label)
        counts[name] = counts.get(name, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))[0] if counts else ""


def _score(group: Sequence[View], mean_vote: float) -> float:
    """A within-query ranking key. Not a probability, not comparable across queries.

    Geometric mean of three independent signals -- how many views support the
    match, how strongly the surviving points are agreed on, and how sure the VLM
    was. `_cluster_confidence` in `fusion.py` combines its terms the same way and
    for the same reason: a match that fails any one of them must not be rescued
    by the other two.
    """
    view_score = 1.0 - np.exp(-len(group) / 2.0)
    vlm_score = float(np.mean([v.vlm_confidence for v in group])) if group else 0.0
    terms = (max(view_score, 1e-6), max(mean_vote, 1e-6), max(vlm_score, 1e-6))
    return float(np.clip(np.prod(terms) ** (1 / 3), 0.0, 1.0))
```

Add a placeholder `_vlm_views` so the module imports cleanly; Task 9 fills it in. Task 8's tests never reach it — a phrase with no cached match and no detector returns before this point — so it is staged, not stubbed:

```python
def _vlm_views(recon, doc, phrase, detector, notes: list[str]) -> list[View]:
    raise NotImplementedError("targeted detection lands in Task 9")
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_room.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/room3d/query.py tests/test_query_room.py
git commit -m "Add query_room: carve, fit and rank matches

Returns matches best-first with the vote statistics that produced them, so
min_vote is visible rather than magic. A match found in 2D but carved away
entirely is returned as unsupported, because that is a different answer from
the object not being there.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Targeted detection — the VLM path

Fill in `_vlm_views` and give `GeminiDetector` a way to look for one named thing.

**Files:**
- Modify: `src/room3d/vlm.py`
- Modify: `src/room3d/query.py:_vlm_views`
- Test: `tests/test_query_vlm.py`

**Interfaces:**
- Consumes: `GeminiDetector._call_with_retry`, `GeminiDetector._parse`, `GeminiDetector._to_part` (existing private helpers).
- Produces: `GeminiDetector.locate(image, phrase) -> list[Detection]`.

**Frames searched:** those listed in `observations_doc["frames_labeled"]`, falling back to every frame in the reconstruction. Those are the frames already chosen for coverage; re-deriving the selection would spend calls to reach the same answer.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_query_vlm.py`:

```python
"""Gate 8d: the VLM path, with the network stubbed out.

The rule under test is when Gemini is called at all. Calling it when the cache
could have answered wastes a scarce quota; not calling it when the cache cannot
answer produces a confident wrong "not found".
"""

import json

import numpy as np

from room3d.artifacts import save_frames_npz
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from room3d.query import query_room
from room3d.vlm import Detection
from test_boxfit_wiring import labelled_session, sofa_detection


class StubDetector:
    """Returns the sofa's box for any phrase, and counts every call."""

    def __init__(self, detection=None):
        self.calls = []
        self._detection = detection

    def locate(self, image, phrase):
        self.calls.append(phrase)
        det = self._detection if self._detection is not None else sofa_detection()
        return [Detection(label=phrase, box_2d=det.box_2d, confidence=0.85)]


def room_on_disk(tmp_path):
    session = labelled_session()
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)
    save_frames_npz(tmp_path / "frames.npz", session.recon)
    (tmp_path / "observations.json").write_text(
        json.dumps(build_observations_doc(session, "room"))
    )
    (tmp_path / "objects.json").write_text(json.dumps({"room": "room", "objects": []}))
    return tmp_path


def test_a_cached_hit_never_calls_the_detector(tmp_path):
    detector = StubDetector()
    result = query_room(room_on_disk(tmp_path), "sofa", detector=detector, verbose=False)
    assert detector.calls == []
    assert result.source == "cache"


def test_a_cached_miss_calls_the_detector(tmp_path):
    detector = StubDetector()
    result = query_room(
        room_on_disk(tmp_path), "aquarium", detector=detector, verbose=False
    )
    assert detector.calls
    assert result.source == "vlm"


def test_a_qualified_phrase_goes_to_the_detector_even_though_its_noun_is_cached(tmp_path):
    """"sofa" is cached; "the sofa by the window" is a spatial relation the
    labels cannot evaluate, so answering it from the cache would answer a
    different question."""
    detector = StubDetector()
    query_room(
        room_on_disk(tmp_path), "the sofa by the window",
        detector=detector, verbose=False,
    )
    assert detector.calls
    assert set(detector.calls) == {"the sofa by the window"}


def test_force_bypasses_a_cached_hit(tmp_path):
    detector = StubDetector()
    result = query_room(
        room_on_disk(tmp_path), "sofa", detector=detector, force=True, verbose=False
    )
    assert detector.calls
    assert result.source == "vlm"


def test_the_detector_is_asked_once_per_labelled_frame(tmp_path):
    detector = StubDetector()
    room = room_on_disk(tmp_path)
    frames = json.loads((room / "observations.json").read_text())["frames_labeled"]
    query_room(room, "aquarium", detector=detector, verbose=False)
    assert len(detector.calls) == len(frames)


def test_vlm_views_produce_a_usable_box(tmp_path):
    detector = StubDetector()
    result = query_room(
        room_on_disk(tmp_path), "sofa", detector=detector, force=True, verbose=False
    )
    assert result.matches
    assert result.matches[0].obb is not None


def test_a_detector_error_on_one_frame_does_not_kill_the_query(tmp_path):
    """Gemini's free tier 429s often enough that all-or-nothing would routinely
    return nothing, which is indistinguishable from the object being absent."""

    class Flaky:
        def __init__(self):
            self.calls = []

        def locate(self, image, phrase):
            self.calls.append(phrase)
            if len(self.calls) == 1:
                raise RuntimeError("429 rate limited")
            det = sofa_detection()
            return [Detection(label=phrase, box_2d=det.box_2d, confidence=0.85)]

    detector = Flaky()
    result = query_room(
        room_on_disk(tmp_path), "aquarium", detector=detector, verbose=False
    )
    assert len(detector.calls) > 1               # it kept going after the failure
    assert result.matches
    assert any("429" in n for n in result.notes)


def test_locate_prompt_names_the_phrase_and_permits_an_empty_answer():
    from room3d.vlm import LOCATE_PROMPT

    prompt = LOCATE_PROMPT.format(phrase="office chair")
    assert "office chair" in prompt
    assert "empty" in prompt.lower() or "none" in prompt.lower()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_vlm.py -q`
Expected: `NotImplementedError: targeted detection lands in Task 9` (and an ImportError on `LOCATE_PROMPT`).

- [ ] **Step 3: Add `LOCATE_PROMPT` and `locate` to `src/room3d/vlm.py`**

After `DETECTION_PROMPT`:

```python
LOCATE_PROMPT = """\
You are looking at one frame of a walkthrough of a single room.

Find every physically distinct object in this image matching this description:

    {phrase}

Rules:
- One entry per physical object. If two things match the description, return two.
- If the description mentions a position or a relationship ("by the window", "the
  left one"), only return objects that actually satisfy it.
- If nothing in this image matches, return an empty list. An empty list is a
  correct and useful answer; a guess is not.
- Set `label` to the description you were given, verbatim.
- `box_2d` must be [ymin, xmin, ymax, xmax], normalised 0-1000.
- `confidence` is your own 0-1 certainty that this object matches the description.

Return at most 5 objects."""
```

And as a method on `GeminiDetector`, next to `detect`:

```python
    def locate(self, image: np.ndarray | Path | str, phrase: str) -> list[Detection]:
        """Find one named thing, rather than surveying everything.

        Two differences from `detect` that matter. The model is told what to look
        for, so it can resolve a description the cached labels cannot -- "the
        couch by the window" is a spatial relation, and labels do not carry
        those. And it is told that finding nothing is a correct answer, because
        the alternative is a model that always returns something.
        """
        payload = self._to_part(image)
        raw = self._call_with_retry(LOCATE_PROMPT.format(phrase=phrase), payload)
        return self._parse(raw)
```

- [ ] **Step 4: Implement `_vlm_views` in `src/room3d/query.py`**

Replace the placeholder:

```python
def _vlm_views(recon, doc, phrase, detector, notes: list[str]) -> list[View]:
    """Ask the detector for `phrase` in every frame the room was labelled on.

    Those frames were already chosen for coverage by `select_covering_frames`;
    re-deriving the selection here would spend calls to arrive at the same list.

    One frame failing must not lose the other eleven. Gemini's free tier returns
    429 often enough that an all-or-nothing query would frequently return
    nothing at all, which is indistinguishable from the object being absent.
    """
    from .projection import descale_box

    frames = [int(i) for i in doc.get("frames_labeled", [])]
    frames = [i for i in frames if i < recon.n_frames] or list(range(recon.n_frames))

    height, width = recon.image_hw
    views: list[View] = []
    failures = 0

    for frame_idx in frames:
        try:
            detections = detector.locate(recon.images[frame_idx], phrase)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not lose the rest
            failures += 1
            notes.append(f"frame {frame_idx} failed: {exc}")
            continue

        for det in detections:
            views.append(
                View(
                    frame_idx=frame_idx,
                    box_px=descale_box(det.box_2d, height, width),
                    label=det.label or phrase,
                    vlm_confidence=float(det.confidence),
                )
            )

    if failures:
        notes.append(f"{failures} of {len(frames)} frames failed to be searched")
    if not views:
        notes.append(f"the detector found nothing matching {phrase!r} in any frame")
    return views
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_vlm.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, including `tests/test_vlm.py`.

- [ ] **Step 7: Commit**

```bash
git add src/room3d/vlm.py src/room3d/query.py tests/test_query_vlm.py
git commit -m "Add targeted detection for the query miss path

GeminiDetector.locate asks for one named thing and is told that finding
nothing is a correct answer. A frame that 429s is noted and skipped rather
than losing the other eleven.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Committing a match

Promote a match into `objects.json`, deleting the duplicate entries it subsumes.

**Files:**
- Modify: `src/room3d/query.py`
- Test: `tests/test_query_commit.py`

**Interfaces:**
- Consumes: `QueryMatch`, `room3d.fusion.ObjectRecord`.
- Produces: `commit_match(room_dir, match, *, verbose=True) -> dict` with keys `object_id`, `removed`, `n_objects`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_query_commit.py`:

```python
"""Gate 8e: promoting a query result into the room.

This is the step that fixes duplicates. A query that finds one couch where the
object list holds two must be able to say so destructively -- with a backup,
because a vague phrase must never cost you a labelled room.
"""

import json

import numpy as np
import pytest

from room3d.artifacts import save_frames_npz
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from room3d.query import commit_match, query_room
from test_boxfit_wiring import labelled_session


def room_on_disk(tmp_path, extra_objects=()):
    session = labelled_session()
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)

    save_frames_npz(tmp_path / "frames.npz", session.recon)
    (tmp_path / "observations.json").write_text(
        json.dumps(build_observations_doc(session, "room"))
    )
    objects = [o.as_dict() for o in session.objects] + list(extra_objects)
    (tmp_path / "objects.json").write_text(
        json.dumps({
            "room": "room", "units": "meters", "scale_verified": True,
            "n_frames_labeled": 4, "objects": objects,
        })
    )
    return tmp_path


def decoy(object_id="obj_099"):
    return {
        "id": object_id, "label": "lamp", "aliases": [],
        "centroid": [9.0, 9.0, 9.0],
        "obb": {"center": [9.0, 9.0, 9.0], "extent": [1.0, 1.0, 1.0],
                "R": np.eye(3).tolist()},
        "confidence": 0.5, "n_observations": 1, "seen_in": [0],
        "observation_ids": [],
    }


def test_committing_writes_the_match_into_objects_json(tmp_path):
    room = room_on_disk(tmp_path)
    match = query_room(room, "sofa", verbose=False).matches[0]
    commit_match(room, match, verbose=False)

    written = json.loads((room / "objects.json").read_text())["objects"]
    entry = next(o for o in written if o["label"] == "sofa")
    assert np.allclose(entry["obb"]["extent"], match.obb.extent)


def test_committing_removes_the_objects_the_match_absorbed(tmp_path):
    room = room_on_disk(tmp_path, extra_objects=[decoy()])
    before = len(json.loads((room / "objects.json").read_text())["objects"])

    match = query_room(room, "sofa", verbose=False).matches[0]
    assert match.absorbed_object_ids == ["obj_000"]
    result = commit_match(room, match, verbose=False)

    after = json.loads((room / "objects.json").read_text())["objects"]
    assert len(after) == before                      # one removed, one added
    assert result["removed"] == ["obj_000"]


def test_committing_leaves_unrelated_objects_alone(tmp_path):
    room = room_on_disk(tmp_path, extra_objects=[decoy()])
    match = query_room(room, "sofa", verbose=False).matches[0]
    commit_match(room, match, verbose=False)

    written = json.loads((room / "objects.json").read_text())["objects"]
    assert any(o["id"] == "obj_099" for o in written)


def test_committing_backs_up_the_previous_objects_file(tmp_path):
    room = room_on_disk(tmp_path)
    before = json.loads((room / "objects.json").read_text())

    match = query_room(room, "sofa", verbose=False).matches[0]
    commit_match(room, match, verbose=False)

    assert json.loads((room / "objects.prev.json").read_text()) == before


def test_committing_preserves_run_metadata(tmp_path):
    room = room_on_disk(tmp_path)
    match = query_room(room, "sofa", verbose=False).matches[0]
    commit_match(room, match, verbose=False)

    written = json.loads((room / "objects.json").read_text())
    assert written["scale_verified"] is True
    assert written["room"] == "room"


def test_the_committed_object_reuses_an_absorbed_id_so_ids_stay_stable(tmp_path):
    room = room_on_disk(tmp_path)
    match = query_room(room, "sofa", verbose=False).matches[0]
    assert commit_match(room, match, verbose=False)["object_id"] == "obj_000"


def test_a_match_absorbing_nothing_gets_a_fresh_unused_id(tmp_path):
    room = room_on_disk(tmp_path, extra_objects=[decoy("obj_005")])
    match = query_room(room, "sofa", verbose=False).matches[0]
    match.absorbed_object_ids = []

    new_id = commit_match(room, match, verbose=False)["object_id"]
    existing = {o["id"] for o in json.loads((room / "objects.json").read_text())["objects"]}
    assert new_id in existing
    assert new_id not in {"obj_000", "obj_005"}


def test_an_unsupported_match_cannot_be_committed(tmp_path):
    room = room_on_disk(tmp_path)
    result = query_room(room, "sofa", config_overrides={"min_vote": 1.01}, verbose=False)
    with pytest.raises(ValueError, match="unsupported"):
        commit_match(room, result.matches[0], verbose=False)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_commit.py -q`
Expected: `ImportError: cannot import name 'commit_match'`.

- [ ] **Step 3: Write the implementation**

Append to `src/room3d/query.py`:

```python
def commit_match(room_dir: str | Path, match: QueryMatch, *, verbose: bool = True) -> dict:
    """Write a match into `objects.json`, replacing what it subsumes.

    This is the step that fixes duplicates. A query that finds one couch where
    the object list holds two removes both and writes one, on the strength of the
    cross-view evidence rather than a threshold that happened to work.

    Destructive, so it backs up first: a vague phrase must never cost a labelled
    room. Same `.prev.json` convention as `refit.py`.
    """
    from .fusion import ObjectRecord

    if not match.supported or match.obb is None:
        raise ValueError(
            "this match is unsupported -- it was found in 2D but no 3D points "
            "survived carving, so there is no box to commit"
        )

    room_dir = Path(room_dir)
    path = room_dir / "objects.json"
    doc = json.loads(path.read_text()) if path.exists() else {"objects": []}
    objects = list(doc.get("objects", []))

    absorbed = set(match.absorbed_object_ids)
    kept = [o for o in objects if o.get("id") not in absorbed]
    removed = [o["id"] for o in objects if o.get("id") in absorbed]

    # Reuse an absorbed id where there is one: the new object *is* the old one,
    # better measured, and stable ids keep any external references working.
    if match.absorbed_object_ids:
        object_id = sorted(match.absorbed_object_ids)[0]
    else:
        used = {o.get("id", "") for o in kept}
        n = 0
        while f"obj_{n:03d}" in used:
            n += 1
        object_id = f"obj_{n:03d}"

    labels = {normalize_label(v.label) for v in match.views}
    record = ObjectRecord(
        id=object_id,
        label=match.label,
        aliases=sorted(labels - {normalize_label(match.label)}),
        centroid=np.asarray(match.obb.center, dtype=float),
        obb=match.obb,
        confidence=match.score,
        n_observations=len(match.views),
        seen_in=sorted({v.frame_idx for v in match.views}),
        observation_ids=sorted(
            v.observation_id for v in match.views if v.observation_id is not None
        ),
    )

    path.with_suffix(".prev.json").write_text(json.dumps(doc, indent=2))
    doc["objects"] = kept + [record.as_dict()]
    path.write_text(json.dumps(doc, indent=2))

    if verbose:
        print(f"[query] committed {object_id} ({match.label}); "
              f"removed {len(removed)} absorbed object(s)")

    return {"object_id": object_id, "removed": removed, "n_objects": len(doc["objects"])}
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_commit.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 5: Commit**

```bash
git add src/room3d/query.py tests/test_query_commit.py
git commit -m "Add commit_match: promote a query result, absorbing duplicates

One couch listed twice becomes one couch, on cross-view evidence rather than
a tuned threshold. Backs up to objects.prev.json first, because a vague
phrase must never cost a labelled room.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: The `query` CLI verb

**Files:**
- Modify: `src/room3d/cli.py`
- Modify: `README.md`
- Test: `tests/test_query_cli.py`

**Interfaces:**
- Consumes: `query_room`, `commit_match`, `load_config`.
- Produces: `cmd_query(args) -> int`, and the `query` subparser.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_query_cli.py`:

```python
"""Gate 8f: the CLI verb.

Read-only unless --commit is given. That is the property that makes a vague
query free to try.
"""

import json

import pytest

from room3d.artifacts import save_frames_npz
from room3d.cli import main
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from test_boxfit_wiring import labelled_session


@pytest.fixture
def room(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out" / "demo"
    out.mkdir(parents=True)

    session = labelled_session()
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)

    save_frames_npz(out / "frames.npz", session.recon)
    (out / "observations.json").write_text(
        json.dumps(build_observations_doc(session, "demo"))
    )
    (out / "objects.json").write_text(
        json.dumps({"room": "demo", "units": "meters", "scale_verified": False,
                    "n_frames_labeled": 4,
                    "objects": [o.as_dict() for o in session.objects]})
    )
    return out


def test_query_returns_zero_and_prints_a_match(room, capsys):
    assert main(["query", "--room", "demo", "sofa"]) == 0
    assert "sofa" in capsys.readouterr().out


def test_query_does_not_modify_the_room(room):
    before = (room / "objects.json").read_text()
    main(["query", "--room", "demo", "sofa"])
    assert (room / "objects.json").read_text() == before


def test_json_output_parses(room, capsys):
    main(["query", "--room", "demo", "sofa", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["phrase"] == "sofa"
    assert payload["matches"]


def test_commit_writes_the_match(room):
    before = (room / "objects.json").read_text()
    assert main(["query", "--room", "demo", "sofa", "--commit", "1"]) == 0
    assert (room / "objects.json").read_text() != before
    assert (room / "objects.prev.json").exists()


def test_commit_is_one_indexed_and_rejects_a_bad_index(room, capsys):
    assert main(["query", "--room", "demo", "sofa", "--commit", "0"]) == 1
    assert "1-indexed" in capsys.readouterr().err


def test_a_missing_room_reports_an_error_rather_than_a_traceback(room, capsys):
    assert main(["query", "--room", "nope", "sofa"]) == 1
    assert "not found" in capsys.readouterr().err


def test_min_vote_can_be_overridden_from_the_command_line(room, capsys):
    main(["query", "--room", "demo", "sofa", "--min-vote", "0.9", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["matches"][0]["vote_stats"]["min_vote"] == 0.9


def test_no_match_is_reported_and_is_not_an_error(room, capsys):
    assert main(["query", "--room", "demo", "aquarium"]) == 0
    assert "no detector" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_cli.py -q`
Expected: `argparse` error — `invalid choice: 'query'` — surfacing as `SystemExit`.

- [ ] **Step 3: Add `cmd_query` to `src/room3d/cli.py`**

Insert before `cmd_view`:

```python
def cmd_query(args) -> int:
    from .query import commit_match, query_room

    out = _out_dir(args.room)
    cfg = load_config(args.config).query
    overrides = {}
    if args.min_vote is not None:
        overrides["min_vote"] = args.min_vote

    detector = None
    if args.force or args.detector:
        detector = _build_detector(args)

    try:
        result = query_room(
            out, args.phrase,
            config=cfg, config_overrides=overrides or None,
            detector=detector, force=args.force,
            verbose=not args.json,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    elif not result.matches:
        for note in result.notes:
            print(f"[query] {note}")

    if args.commit is not None:
        if args.commit < 1 or args.commit > len(result.matches):
            print(f"error: --commit is 1-indexed; this query returned "
                  f"{len(result.matches)} match(es)", file=sys.stderr)
            return 1
        try:
            commit_match(out, result.matches[args.commit - 1], verbose=not args.json)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    return 0


def _build_detector(args):
    """A detector only when one is actually needed -- building it loads .env."""
    from .vlm import GeminiDetector, load_env, resolve_model

    load_env()
    cfg = load_config(args.config).label
    return GeminiDetector(
        model=resolve_model(cfg.model),
        request_masks=False,
        max_retries=cfg.max_retries,
        min_interval_s=60.0 / max(cfg.max_rpm, 1),
    )
```

Add `import json` to the top of `cli.py` alongside the existing imports.

- [ ] **Step 4: Register the subparser**

Insert before the `view` subparser:

```python
    q = sub.add_parser("query", help="name an object, get its 3D box (no reconstruction)")
    q.add_argument("phrase", help='e.g. "couch" or "the couch by the window"')
    q.add_argument("--room", required=True)
    q.add_argument("--force", action="store_true",
                   help="ignore the cached detections and ask the VLM")
    q.add_argument("--detector", action="store_true",
                   help="allow VLM calls when the cache cannot answer")
    q.add_argument("--commit", type=int, default=None, metavar="N",
                   help="promote match N (1-indexed) into objects.json")
    q.add_argument("--min-vote", type=float, default=None,
                   help="fraction of views that must agree (default 0.6)")
    q.add_argument("--json", action="store_true", help="machine-readable output")
    q.set_defaults(func=cmd_query)
```

- [ ] **Step 5: Update the module docstring at the top of `cli.py`**

```python
    room3d query       --room office "couch"     # name a thing, get its box
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_cli.py -q`
Expected: PASS (every test in the file).

- [ ] **Step 7: Update `README.md`**

In the Usage code block, after the `refit` line:

```bash
# name one object and get its box, carved by cross-view agreement
uv run room3d query --room office "couch"
uv run room3d query --room office "couch" --commit 1   # replace the duplicates
```

Add to the Layout tree, after `refit.py`:

```
├── camera.py        world -> pixel, and whether a point was actually seen
├── consensus.py   ★ cross-view voting: carve boxes, decide identity  (pure, tested)
├── query.py         name an object, get its box; optionally commit it
```

Add a section after "How the 3D boxes are fit":

```markdown
## Querying one object

`room3d query --room living "couch"` answers from the detections already on
disk. It costs nothing and calls nothing, because the expensive part of a run —
good 2D boxes — is already there.

What it adds is other cameras. An axis-aligned 2D box around a non-rectangular
object contains things that are not the object; a `cabinet` box also holds the
sofa in front of it. That information is not in the frame, but it is in the
other frames: clutter sits at a different depth, so under parallax it projects
*outside* the box seen from another angle while the object projects inside every
one of them. `consensus.carve` counts those agreements and keeps what survives.

Two rules make the count honest. A frame that could not see a point does not
vote on it — occlusion is not disagreement. And a point does not vote for
itself, since a point harvested from frame 6's box is inside frame 6's box by
construction.

The same vote answers the other question. If object A's points land inside
object B's boxes and B's inside A's, they are one object listed twice — which is
why `--commit` can replace several duplicate entries with one, on evidence
rather than on a tuned threshold.

It does not separate objects that physically touch: a sofa arm resting against a
cabinet is genuinely one connected mass of points, and no geometric test splits
it. That needs a segmentation model and is deliberately not implemented.
```

- [ ] **Step 8: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 9: Verify against the real room**

Run: `.venv/Scripts/python.exe -m room3d.cli query --room LR_2 "couch"`
Expected: at least one match, with an extent and vote statistics printed. Compare the extent against `out/LR_2/objects.json`; carving should report `kept_frac` below 1.0. This is a smoke check, not an assertion — record what you see in the commit message.

- [ ] **Step 10: Commit**

```bash
git add src/room3d/cli.py README.md tests/test_query_cli.py
git commit -m "Add the query CLI verb and document it

Read-only unless --commit is given, which is what makes a vague phrase free
to try. --commit is 1-indexed to match the printed numbering.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Done when

- `.venv/Scripts/python.exe -m pytest -q` passes with roughly 95 new tests on top of the existing 265.
- `room3d query --room LR_2 "couch"` returns a match with vote statistics and a levelled box.
- `room3d query --room LR_2 "couch" --commit 1` replaces the duplicate couch entries with one and writes `objects.prev.json`.
- No new entries in `pyproject.toml`.

---

### Task 12: The certainty gate

Added mid-execution at the user's request: *"I wouldn't mind losing some object labels if there is no certainty of its position."* Precision over recall.

The engine already computes what certainty means — how many views support a match, and how strongly the surviving points are agreed on. What it does not yet do is act on it: `query_room` returns everything it found and leaves the judgement to the reader.

**Files:**
- Modify: `src/room3d/config.py` (`QueryConfig`, `LabelConfig`)
- Modify: `src/room3d/query.py`
- Modify: `src/room3d/refit.py`
- Modify: `src/room3d/cli.py`
- Modify: `README.md`
- Test: `tests/test_certainty.py`

**Interfaces:**
- Consumes: `QueryMatch` and `commit_match` (Tasks 8, 10), `refit_room` (existing module).
- Produces:
  - `QueryConfig.min_views: int = 2`, `QueryConfig.min_mean_vote: float = 0.5`
  - `LabelConfig.min_observations: int = 1`
  - `query.filter_by_certainty(matches, *, min_views, min_mean_vote) -> tuple[list[QueryMatch], list[QueryMatch]]` returning `(kept, dropped)`
  - `commit_match(room_dir, match, *, force=False, verbose=True)` — refuses a match below the gate unless forced

**The governing principle: nothing vanishes silently.** Every path that drops something reports how many and why. The user asked to lose uncertain labels — not to lose the knowledge that they existed. A filter that quietly halves the object list is indistinguishable from a bug.

**Two different defaults, deliberately.** A query is a pointed question whose answer should be trustworthy, so `min_views=2` is the default there: one view can be cross-checked against nothing, which is exactly the "no certainty of position" case. Bulk `refit` keeps `min_observations=1` and stays lossless, with the switch one flag away — dropping 42 of `LR_2`'s 50 objects should be something the user asks for, not something a config default does to them.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_certainty.py`:

```python
"""Gate 9: losing labels we cannot place, on purpose.

An object whose position is uncertain is worse than a missing one, because a
confident wrong box gets acted on and an absent one does not. What certainty
means here is already computed -- how many views support the match, and how
strongly the surviving points are agreed on -- so this is a filter over
existing evidence, not new machinery.

The governing rule is that nothing vanishes silently: every drop is counted
and returned to the caller.
"""

import json

import numpy as np
import pytest

from room3d.config import LabelConfig, QueryConfig
from room3d.consensus import View
from room3d.projection import OrientedBox
from room3d.query import QueryMatch, commit_match, filter_by_certainty
from room3d.refit import refit_room


def a_match(n_views=3, mean_vote=0.9, supported=True):
    obb = OrientedBox(np.zeros(3), np.ones(3), np.eye(3))
    return QueryMatch(
        label="couch",
        obb=obb if supported else None,
        score=0.8,
        views=[View(i, (0, 0, 10, 10)) for i in range(n_views)],
        n_points=500,
        vote_stats={"mean_vote": mean_vote, "min_vote": 0.6,
                    "n_candidates": 1000, "n_kept": 500, "kept_frac": 0.5},
        supported=supported,
    )


def test_a_single_view_match_is_dropped():
    """One view can be cross-checked against nothing. That IS the uncertain case."""
    kept, dropped = filter_by_certainty([a_match(n_views=1)],
                                        min_views=2, min_mean_vote=0.5)
    assert kept == []
    assert len(dropped) == 1


def test_a_well_supported_match_survives():
    kept, dropped = filter_by_certainty([a_match()], min_views=2, min_mean_vote=0.5)
    assert len(kept) == 1
    assert dropped == []


def test_a_weakly_agreed_match_is_dropped():
    kept, _ = filter_by_certainty([a_match(mean_vote=0.2)],
                                  min_views=2, min_mean_vote=0.5)
    assert kept == []


def test_an_unsupported_match_is_dropped_however_open_the_gate():
    """No 3D points survived carving, so there is no position to be certain of."""
    kept, _ = filter_by_certainty([a_match(supported=False)],
                                  min_views=0, min_mean_vote=0.0)
    assert kept == []


def test_dropped_matches_are_returned_not_discarded():
    _, dropped = filter_by_certainty([a_match(n_views=1)],
                                     min_views=2, min_mean_vote=0.5)
    assert dropped[0].label == "couch"
    assert len(dropped[0].views) == 1


def test_the_gate_can_be_opened_completely():
    kept, dropped = filter_by_certainty([a_match(n_views=1, mean_vote=0.0)],
                                        min_views=0, min_mean_vote=0.0)
    assert len(kept) == 1 and dropped == []


def test_ranking_is_preserved_among_survivors():
    kept, _ = filter_by_certainty([a_match(n_views=5), a_match(n_views=3)],
                                  min_views=2, min_mean_vote=0.5)
    assert [len(m.views) for m in kept] == [5, 3]


# --- the defaults encode the preference ---------------------------------------


def test_query_defaults_require_cross_view_support():
    assert QueryConfig().min_views == 2
    assert QueryConfig().min_mean_vote == 0.5


def test_refit_defaults_stay_lossless():
    """Dropping most of a room is something the user asks for, not a default."""
    assert LabelConfig().min_observations == 1


# --- committing --------------------------------------------------------------


def test_committing_an_uncertain_match_is_refused(tmp_path):
    (tmp_path / "objects.json").write_text(json.dumps({"room": "r", "objects": []}))
    with pytest.raises(ValueError, match="certainty gate"):
        commit_match(tmp_path, a_match(n_views=1), verbose=False)


def test_force_overrides_the_certainty_gate(tmp_path):
    (tmp_path / "objects.json").write_text(json.dumps({"room": "r", "objects": []}))
    assert commit_match(tmp_path, a_match(n_views=1), force=True,
                        verbose=False)["object_id"]


# --- the same trade, applied to a whole room ---------------------------------


def test_refit_can_drop_thinly_supported_objects(tmp_path):
    """The user's actual case: most objects in the sample room are single
    sightings."""
    from test_refit import legacy_room

    room = legacy_room(tmp_path)
    lossless = refit_room(room, verbose=False)

    strict = refit_room(
        room, config=LabelConfig(min_mask_points=10, min_observations=99),
        verbose=False,
    )
    assert strict["n_objects"] == 0
    assert strict["n_dropped_uncertain"] == lossless["n_objects"]


def test_refit_reports_nothing_dropped_by_default(tmp_path):
    from test_refit import legacy_room

    assert refit_room(legacy_room(tmp_path), verbose=False)["n_dropped_uncertain"] == 0
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_certainty.py -q`
Expected: `ImportError: cannot import name 'filter_by_certainty' from 'room3d.query'`.

- [ ] **Step 3: Add the config fields**

In `src/room3d/config.py`, append to `QueryConfig`:

```python
    # --- the certainty gate ---
    # A match nothing could cross-check is not a located object; it is a guess
    # with coordinates attached. Dropping it beats reporting it, because a
    # confident wrong box gets acted on and a missing one does not.
    min_views: int = 2
    min_mean_vote: float = 0.5
```

Append to `LabelConfig`:

```python
    # Bulk relabelling stays lossless by default. Raise it to trade recall for
    # certainty -- on the sample living room, 2 drops every single-sighting
    # object, which is most of them.
    min_observations: int = 1
```

- [ ] **Step 4: Implement the filter in `src/room3d/query.py`**

```python
def filter_by_certainty(
    matches: Sequence[QueryMatch],
    *,
    min_views: int,
    min_mean_vote: float,
) -> tuple[list[QueryMatch], list[QueryMatch]]:
    """Split matches into those we can place and those we cannot: `(kept, dropped)`.

    Both halves are returned because nothing may vanish silently. A filter that
    quietly halves the object list is indistinguishable from a bug, and the
    request was to lose uncertain labels -- not to lose the knowledge that they
    were found.

    A single-view match fails this gate by construction: cross-view agreement is
    the only evidence here that a box is where it claims to be, and one view has
    nothing to be checked against.
    """
    kept: list[QueryMatch] = []
    dropped: list[QueryMatch] = []
    for m in matches:
        certain = (
            m.supported
            and len(m.views) >= min_views
            and float(m.vote_stats.get("mean_vote", 0.0)) >= min_mean_vote
        )
        (kept if certain else dropped).append(m)
    return kept, dropped
```

- [ ] **Step 5: Gate `commit_match`**

Change the signature to `def commit_match(room_dir, match, *, force: bool = False, verbose: bool = True) -> dict:` and add this immediately after the existing `supported` check:

```python
    if not force:
        config = QueryConfig()
        if not filter_by_certainty(
            [match], min_views=config.min_views, min_mean_vote=config.min_mean_vote
        )[0]:
            raise ValueError(
                f"this match is below the certainty gate ({len(match.views)} view(s), "
                f"mean vote {match.vote_stats.get('mean_vote', 0.0):.2f}) -- writing it "
                f"into objects.json would record a position nothing verified. "
                f"Pass force=True to commit it anyway."
            )
```

- [ ] **Step 6: Apply the gate in `cmd_query`**

After the result returns and before printing or committing:

```python
    kept, dropped = result.matches, []
    if not args.all:
        kept, dropped = filter_by_certainty(
            result.matches, min_views=cfg.min_views, min_mean_vote=cfg.min_mean_vote
        )
    if dropped and not args.json:
        print(f"[query] {len(dropped)} match(es) hidden as too uncertain to place "
              f"(need {cfg.min_views}+ views agreeing at {cfg.min_mean_vote:.2f}); "
              f"--all shows them")
```

`--commit N` must index into `kept`, not `result.matches`, and pass `force=args.force_commit`. Update the bounds message to report `len(kept)`.

Add the flags to the `query` subparser:

```python
    q.add_argument("--all", action="store_true",
                   help="include matches too uncertain to place")
    q.add_argument("--force-commit", action="store_true",
                   help="commit even if the match is below the certainty gate")
```

- [ ] **Step 7: Apply `min_observations` in `src/room3d/refit.py`**

In `refit_room`, immediately after `cluster_observations` returns:

```python
    dropped = [o for o in objects if o.n_observations < config.min_observations]
    objects = [o for o in objects if o.n_observations >= config.min_observations]
```

Add `"n_dropped_uncertain": len(dropped)` to the returned summary, name it in the verbose line, and add `--min-observations` to the `refit` subparser wired into `config.min_observations`.

- [ ] **Step 8: Run the tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_certainty.py -q`
Then: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (every test in the file), then the whole suite green.

- [ ] **Step 9: Document it in `README.md`**

Append to the "Querying one object" section:

```markdown
By default a query reports only objects it can actually place: at least two views
that agree. One view can be cross-checked against nothing, so its box is a guess
with coordinates attached — and a confident wrong box gets acted on, while a
missing one does not. `--all` shows what was hidden, and the count is always
printed, because a filter that silently halves the object list is
indistinguishable from a bug.

`room3d refit --min-observations 2` applies the same trade to a whole room.
```

- [ ] **Step 10: Commit**

```bash
git add src/room3d/config.py src/room3d/query.py src/room3d/refit.py src/room3d/cli.py tests/test_certainty.py README.md
git commit -m "Drop matches whose position nothing verified"
```

(Full message body should explain the trade; end with the required `Co-Authored-By` trailer.)
