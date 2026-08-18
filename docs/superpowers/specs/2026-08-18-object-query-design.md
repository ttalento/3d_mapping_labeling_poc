# Object query: name a thing, get its box

**Date:** 2026-08-18
**Status:** approved, not yet implemented

## The problem

Two complaints about the labelled output, with different causes:

1. **3D boxes are inaccurate.** The 2D detections are good; the boxes derived from
   them are not. Two reasons. An axis-aligned 2D box around a non-rectangular
   object contains a great deal of non-object — a `cabinet` box that also holds
   the sofa in front of it and a strip of floor. And only the visible surface of
   an object is ever reconstructed, so its box reaches to the front face and no
   further.
2. **One object is listed several times.** The living room reports 47 objects,
   two of them the same couch. Fusion decides identity from centroid distance,
   which is a proxy for the question rather than the question.

Both are downstream of geometry that is already on disk. Neither needs the
reconstruction recomputed.

## The insight

Poses, intrinsics and per-frame pointmaps are mutually consistent: projecting
`pts3d[i]` back through `poses[i]` and `intrinsics[i]` round-trips at **0.000 px
median error** (measured on `out/LR_2`). So any 3D point can be projected into
any frame and asked: *does it land inside that frame's 2D box?*

That single question answers both complaints:

- Point it at one object's boxes and it **carves** — clutter that sits at a
  different depth projects outside the box in other views under parallax, and
  gets voted out.
- Point it at *two* candidate objects and it **identifies** — if A's points land
  inside B's boxes and B's inside A's, they are one object listed twice.

Measured on `out/LR_2` with cached boxes and no LLM calls:

| object | views | extent before | extent after carving | volume before → after |
|---|---|---|---|---|
| cabinet | 2 | 0.40 × 0.49 × 1.26 | 0.29 × 0.47 × 0.91 | 0.247 → 0.124 |
| window | 3 | 0.30 × 0.58 × 0.77 | 0.24 × 0.53 × 0.55 | 0.134 → 0.070 |
| office chair | 3 | 0.87 × 0.39 × 0.38 | 0.45 × 0.20 × 0.91 | 0.129 → 0.082 |
| sofa | 3 | 1.10 × 0.39 × 0.67 | 1.10 × 0.39 × 0.65 | 0.287 → 0.279 |

Compare **volumes, not components**: the fit re-derives yaw from whatever points
survive, so a carved box's axes are not the uncarved box's axes and the three
numbers do not correspond one to one. Adding largest-component selection takes
the office chair further, to 0.73 × 0.19 × 0.28 (volume 0.039), and the chair
from 0.099 to 0.063.

## Scope

**In:** a query subsystem that takes a phrase, returns ranked 3D boxes with the
evidence behind them, and can optionally promote a result into `objects.json`,
absorbing the duplicate entries it subsumes.

**Out:** segmentation models (SAM2, Gemini masks), recovery of unseen geometry
(category size priors, symmetry completion), and any change to the
reconstruction stage. Decided against: geometry-derived masks are enough for
now, and they add no dependency.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Query source | Cached first, VLM on miss, `--force` to bypass | The expensive part — good 2D boxes — is already on disk |
| Query output | Answer, then optionally commit | A bad query costs nothing; a good one cleans up duplicates |
| Box-from-boxes | Cross-view point voting + occlusion test | Measured to work; no grid, no new dependency |
| Object shape | Geometry-derived masks, no segmentation model | Zero new dependencies; the mask falls out of the carving for free |

## Architecture

```
phrase ──▶ views ──▶ instances ──▶ carve ──▶ box ──▶ ranked matches ──▶ commit?
          (cache      (mutual      (vote +   (existing
           or VLM)     agreement)   occlude)  gravity fit)
```

A **view** is `(frame_idx, box_px)` — the cached artifact that is already
trusted. Everything downstream consumes views, so the cached and VLM paths share
one code path.

### `src/room3d/camera.py` (new, ~70 lines, pure)

The direction the repo cannot currently go. Everything today is pixel → world;
this is world → pixel.

```python
project_to_frame(points, pose_c2w, K, hw) -> (uv, in_view)
visible_in_frame(points, recon, frame, *, tol) -> bool mask
```

`visible_in_frame` is the occlusion test: compare a point's distance from the
camera against what `pts3d[frame]` records at the pixel it lands on. Farther than
that surface means it was hidden, so **the frame gets no vote** rather than a
wrong one. Conflating "disagrees" with "could not see" is what destroyed the
`speaker` in the first probe — 427 points survived a unanimity rule that should
never have been asked of frames with no opinion.

A separate module rather than an addition to `projection.py`: that file is ~470
lines and is *about* the opposite direction.

### `src/room3d/consensus.py` (new, ~120 lines, pure)

```python
vote(points, views, recon) -> (agree, testable)          # per-point counts
consistent(points, views, recon, *, min_vote) -> bool mask
consistency_mask(frame, views, recon) -> (H, W) bool     # the viewer overlay
largest_component(points, *, voxel) -> bool mask         # scipy.ndimage.label
agreement_between(views_a, views_b, recon) -> float      # the identity test, 0..1
carve(views, recon, *, min_vote, keep_largest) -> points
```

`min_vote` is a **fraction of frames that could see the point**, not a veto.
Unanimity is wrong: it destroys small objects seen by few cameras.

`agreement_between` is **symmetric by construction**: take the fraction of A's
points that land inside B's boxes where B could see them, the same for B into A,
and return the smaller of the two. The minimum, not the mean — a small object
sitting inside a large one scores high in one direction and must not be merged on
that evidence alone.

`consistency_mask` is the same computation transposed — score pixels rather than
pooled points. That is where the per-frame mask comes from, at no extra cost.

`carve` returns points. The box fit is unchanged: `fit_gravity_aligned_box` then
`snap_to_floor`, both already in `projection.py`. **Carving replaces the point
selection, not the fit.**

### `src/room3d/query.py` (new)

**Stage 1 — phrase → views.** Cached: match the phrase against labels in
`observations.json` via `normalize_label` and the existing `SynonymResolver`,
which already knows couch ≡ sofa and can consult the LLM when the static table
misses. On a miss or with `--force`: `GeminiDetector.locate(image, phrase)`, a
new targeted prompt returning 0..N boxes for that phrase across the labelled
frames.

A phrase carrying a qualifier the cache cannot evaluate — "the couch **by the
window**" — counts as a **miss** and falls through to the VLM. Labels alone
cannot resolve spatial language, and answering from the label anyway would
silently answer a different question.

**Stage 2 — views → instances.** "chair" matches eight detections that are four
chairs. Grouping uses `agreement_between`, not a second algorithm. The primitive
lives in `consensus.py`; the greedy grouping policy lives here.

**Stage 3 — carve each instance**, then fit.

**Stage 4 — rank and return.**

```python
@dataclass
class QueryMatch:
    label: str
    obb: OrientedBox
    score: float                      # ranking key, see below
    views: list[View]
    n_points: int
    vote_stats: dict                  # so min_vote is visible, not magic
    absorbed_object_ids: list[str]
```

`score` ranks matches within one query; it is not comparable across queries and
must not be presented as a probability. It is the geometric mean of three terms
already used by `_cluster_confidence` in `fusion.py` — how many views support the
match, the mean vote fraction of its surviving points, and the mean VLM
confidence of its detections — so that a match failing any one of them cannot be
rescued by the other two.

**Commit** builds an `ObjectRecord` from a match, deletes every object named in
`absorbed_object_ids`, and writes with a `.prev.json` backup, reusing the pattern
in `refit.py`. `--commit N` is **1-indexed**, matching the numbering the CLI
prints.

### `QueryConfig` in `config.py`

| Field | Default | Meaning |
|---|---|---|
| `min_vote` | 0.6 | fraction of frames that could see the point which must agree |
| `occlusion_tol` | 0.10 | relative depth slack in the visibility test |
| `keep_largest_component` | `True` | drop spatially disconnected blobs |
| `component_voxel` | 0.04 | connected-component grid resolution |
| `min_instance_agreement` | 0.3 | mutual agreement above which two view-sets are one object |

## Interfaces

```bash
room3d query --room LR_2 "couch"                  # ranked matches, read-only
room3d query --room LR_2 "the couch by the window" --force
room3d query --room LR_2 "cabinet" --commit 1     # promote match 1
room3d query --room LR_2 "chair" --json
```

`POST /api/rooms/{name}/query` with `{phrase, force, commit}` returns the ranked
matches. Each view carries its consistency mask **inline as a base64 PNG** — a
512 × 288 boolean mask compresses to a couple of KB, and inlining keeps the
server stateless with no query-id cache to invalidate.

Viewer: a search box in the Objects panel. Matches highlight in the cloud;
selecting one shows its supporting frames with the consistency mask painted
inside the 2D box, so it is visible whether a `cabinet` box is being counted as
cabinet or as sofa. A "Replace N objects with this" button commits.

## Error handling

- No cached match **and** no API key: report "nothing cached matches, and no
  detector is available" — never an empty result, which reads as "not in the
  room".
- Zero points survive carving: report the match as **unsupported**, with its view
  count. Found in 2D, failed in 3D, and the distinction is the useful part.
- A frame index in `observations.json` beyond the reconstruction: skip and count,
  as `refit.py` already does.

## Testing

TDD throughout, following the existing `tests/` conventions.

- `camera.py` — pixel → world → pixel round-trips to identity; a synthetic
  occluder produces *no vote* rather than a wrong one; points behind the camera
  and off-frame are excluded.
- `consensus.py` — a distractor inside the box in one view and outside in another
  is carved away; **a distractor inside the box in every view survives**, written
  as an explicit test so the limit is documented rather than discovered;
  `min_vote` behaves monotonically; `largest_component` picks the right blob.
- `query.py` — a cached hit never constructs a detector; a cached miss does;
  `--force` bypasses a hit; "chair" yields four instances, not one; commit
  absorbs duplicates and writes the backup.
- Integration on the synthetic levelled room in `test_boxfit_wiring.py`, plus a
  real-data check against `out/LR_2`'s cabinet.

## Implementation order

Two plans, not one. The split is where the language changes:

1. **Geometry and query engine** — `camera.py`, `consensus.py`, `query.py`,
   `QueryConfig`, the CLI verb. Fully testable headless, and it is what makes the
   idea either work or not.
2. **Viewer integration** — the HTTP endpoint, the search box, the mask overlay.
   Worth doing only once (1) has proved out on real rooms.

## Known limits, stated up front

- **Clutter that stays inside the box in every view survives.** A sofa arm
  physically touching the cabinet is one connected mass of points; no geometric
  test separates them. Measured: carving plus component selection handles about
  six of eight multi-view objects in `LR_2`. The real fix is a segmentation
  model, deliberately out of scope.
- **Unseen geometry is still unseen.** Carving tightens a box; it cannot invent
  the back of a sofa.
- **`min_vote` is tuned against one room.** It is exposed in config and its
  statistics are reported per match rather than buried.
