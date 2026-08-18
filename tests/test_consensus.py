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
    but parallax carries it outside the box in frames 0 and 2.

    Baseline 0.5, not the 1.5 used elsewhere in this file: at 1.5 no distractor
    depth nearer than the target is simultaneously inside the flanking cameras'
    field of view and outside their box, given this K and the +-6px box half --
    the two constraints are mutually exclusive at that baseline for any such
    depth. 0.5 is the smallest baseline that keeps both satisfiable.
    """
    recon = make_recon([camera_at(-0.5), camera_at(0.0), camera_at(0.5)])
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
    Counting that as verification inflates every score by one frame.

    Baseline 0.5 -- see the note in
    test_clutter_at_a_different_depth_is_voted_out_by_the_other_views."""
    recon = make_recon([camera_at(-0.5), camera_at(0.0), camera_at(0.5)])
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
