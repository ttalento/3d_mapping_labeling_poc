# room3d

Turn a casual phone walkthrough of a room into a dense 3D point cloud, a camera
trajectory, and a **labeled** scene — objects with approximate 3D positions,
produced by an agentic pipeline rather than manual annotation.

Runs entirely on CPU. No NVIDIA GPU required.

```
video.mp4 ──► frames ──► MASt3R + DUSt3R global alignment ──► scene.ply
                                    │                          trajectory.txt
                                    └──► frames.npz ──► CrewAI + Gemini ──► objects.json
```

## Why not MASt3R-SLAM

The brief specified MASt3R-SLAM. It compiles custom CUDA kernels at install time
and has no CPU path, so it cannot run on a machine without an NVIDIA GPU.

We keep the model and drop only the SLAM layer. `naver/mast3r` ships
`demo_dust3r_ga.py` — "the same demo as in dust3r (+ compatibility for MASt3R
models)" — which runs MASt3R weights through DUSt3R's global aligner. The only
CUDA dependency on that path is an optional RoPE kernel with a pure-PyTorch
fallback in croco's `pos_embed.py`.

**Pose estimation is not dropped.** Global alignment *is* the pose solver:
`get_im_poses()` and `get_focals()` come out of the same optimisation that
produces `get_pts3d()`. What we give up is live tracking and loop closure, which
offline batch work does not need.

## The idea that makes labeling cheap

Global alignment returns, for every input image, a **per-pixel 3D point already
in world frame**. So lifting a 2D detection into 3D is an array lookup —
pixel `(u,v)` in frame `i` is `pts3d[i][v,u]` — not a raycast into a cloud.

That demotes projection from hard to easy and promotes **cross-frame label
fusion** to the real problem: one chair seen in six frames must become one
object, while two different chairs must stay two.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and git.

```bash
git clone --recursive --depth 1 https://github.com/naver/mast3r.git third_party/mast3r
uv python install 3.11
uv sync

# ~2.6 GB checkpoint
curl -o checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

cp .env.example .env      # then add your Gemini API key
```

`third_party/mast3r` bundles dust3r, which bundles croco. Cloning dust3r
separately would put two copies of the same package on `sys.path`.

**`.env` naming.** The key is read from the first of `GEMINI_API_KEY`,
`GOOGLE_API_KEY`, `API_KEY`. The model comes from `GEMINI_MODEL` or `MODEL` and
overrides the config file. Routing prefixes are stripped, so a LiteLLM-style
`google/gemini-3.6-flash` works as well as a bare `gemini-3.6-flash` — passing
the prefixed form straight to the google-genai SDK otherwise fails with a bare
`404 Not Found`, which is a miserable error to diagnose.

## Quickstart on the sample clip

```bash
# ~12 min on CPU: 8 frames from the 10 s clip
uv run room3d run video_examples/20260815_111708_short.mp4 --room living --n-frames 8

# then look at it
uv run room3d view out/living
```

Then the real thing — the **full 40 s clip with more frames**, ~20 min:

```bash
uv run room3d run video_examples/20260815_111708.mp4 --room living_full --n-frames 12
```

**Prefer `--n-frames` over trimming.** Frame count is what costs time; clip
length is what buys room coverage. Sampling 12 frames across the full 40 s sees
the whole room for the same compute as 12 frames from a 10 s slice, which only
ever sees one corner. Trim only when you want a deliberately small test.

Rough CPU cost at 13.5 s/pair, `swin-3`:

| `--n-frames` | pairs | approx wall clock |
|---|---|---|
| 8 | 36 | ~12 min |
| 12 | 60 | ~20 min |
| 16 | 84 | ~26 min |
| 24 | 132 | ~39 min |

Labeling adds a couple of minutes on top and is paced by the Gemini free tier.

## The viewer

```bash
uv run python scripts/fetch_vendor.py     # one-time: downloads three.js
uv run room3d ui --open
```

A local web app at `127.0.0.1:8000` with a persistent object sidebar and five tabs:

| Tab | What it shows |
|---|---|
| **Run** | Pick a video, set frames, start a pipeline run; live log with per-stage progress |
| **Frames** | Keyframes with detection boxes; the selected object highlighted, the rest dimmed |
| **Point cloud** | three.js orbit view, object boxes, camera path, decimation slider |
| **Floor plan** | Top-down room map with clickable object footprints; warns if the room is not levelled |
| **Trajectory** | Camera path, per-frame step distances, and total span |

Select an object in the sidebar and it highlights in whichever view is open.

**The Trajectory tab is the one to look at first.** It surfaces the number that
explains most bad results: total camera span. The bundled `living` run spans
**0.44 m across 8 frames** — that is a pan, not a walk, so there is almost no
parallax, depth is poorly constrained, and fusion cannot match an object between
views. The symptom is 28 objects where the room has maybe 12, most with
`n_observations: 1`. The tab says so in as many words.

### Re-fusion

Clustering is pure arithmetic over the cached observations, so re-running it is
instant and costs no API calls. The sidebar exposes the thresholds live.

One thing worth knowing: the merge radius is `max(floor, scale × mean OBB
diagonal)`, and for furniture the size term usually wins — moving the floor from
0.3 to 1.0 m does nothing at all. The panel therefore reports the **effective**
radius and which term is driving it. On `living`, raising the floor to 3 m
collapses 28 objects into 19.

## Usage

```bash
# everything, end to end
uv run room3d run data/rooms/office/walk.mp4 --room office

# or stage by stage — reconstruction is slow, relabeling should not repeat it
uv run room3d extract     data/rooms/office/walk.mp4 --room office
uv run room3d reconstruct out/office/frames        --room office
uv run room3d label       out/office/frames.npz    --room office
uv run room3d view        out/office

# rebuild the 3D boxes from detections already on disk — no VLM calls
uv run room3d refit       --room office

# only for rooms reconstructed before levelling existed
uv run room3d level       --room office
```

`refit` exists because detection is the only expensive step. Everything after it
is arithmetic over arrays already stored, so improving the geometry never has to
mean paying for detection twice.

Useful flags:

| Flag | Effect |
|---|---|
| `--direct` | run the labeling tools in order without the agent loop |
| `--scale-ref 4.2` | rescale so the largest scene extent is 4.2 m |
| `--scene-graph complete` | O(N²) pairs — better, far slower |
| `--allow-mixed` | photo folders with mixed orientations: keep the majority |
| `--no-label` | stop after reconstruction |
| `--min-confidence 0.5` | (view) hide low-confidence objects |

Frames from one video are always uniform, so `--allow-mixed` only matters for
folders of loose photos. Without it, mixed portrait/landscape input is rejected
up front rather than after half an hour of inference.

### `--direct` is a debugging tool, not a toy

It calls the same tools in the same order, without agents. Agent loops add
failure modes — a model re-running a step, or paraphrasing JSON instead of
returning it — unrelated to whether the geometry is right. When a labeled scene
looks wrong, `--direct` answers "geometry or agent?" in one run.

It is also **much cheaper on quota**, which is not a small point: Gemini's free
tier allows ~15 requests/minute per model, and the crew spends requests on agent
reasoning *as well as* detection. The first crew run here died mid-way with a
429. Both paths now share one `label.max_rpm` budget (default 10), agents are
capped at `max_iter=3`, and the detector takes half the budget in crew mode.
If you hit 429s anyway, lower `max_rpm`.

## Segmentation masks: planned, measured, abandoned

The design called for segmentation masks rather than boxes — a box round a chair
is half wall, and the 3D centroid is a statistic over whichever pixels you pick.
Measured against `gemini-3.5-flash-lite` and `gemini-3.6-flash`:

| Configuration | Result |
|---|---|
| `mask` optional in response schema | omitted entirely — 0 masks in 10 detections |
| `mask` required, base64 PNG requested | reply truncates mid-string → invalid JSON |
| no response schema | masks come back as **COCO RLE, not PNG**; ~65k chars, still truncates |

So `use_masks` defaults to **false**. Boxes plus the depth clustering in
`projection.py` are the reliable path — and clustering is precisely the mechanism
that stops a box-shaped selection dragging the centroid onto the wall behind,
which is the job the mask would have done. The design absorbed the change without
modification because that defence already existed for the depth-bleed case.

Mask support stays behind `use_masks` for a future model. A truncated reply is
now detected via `finish_reason` and reported with an actionable message instead
of being retried four times as a `JSONDecodeError`.

## Measured performance

Core Ultra 7 155H (16C/22T), 31 GB RAM, CPU only, 512 px:

| | |
|---|---|
| Inference | **13.5 s per image pair** |
| Global alignment | ~2.1 it/s (300 iters ≈ 2.2 min at 6 frames) |
| 24-frame room | **~39 min** end to end |

`scene_graph: swin-3` is load-bearing. The `complete` graph is O(N²) — 24 frames
is 552 forward passes, over two hours. A walkthrough is temporally ordered, so a
sliding window is also the *correct* graph, not merely the affordable one.

## Layout

```
src/room3d/
├── frames.py        Stage 1  rotation, blur rejection, near-duplicate rejection
├── reconstruct.py   Stage 2  MASt3R + global alignment -> ply, traj, npz
├── level.py         gravity from poses + floor fit -> +Y up, floor at y=0
├── projection.py    ★ 2D detection -> 3D, gravity-aligned boxes  (pure, tested)
├── fusion.py        ★ observations -> unique objects  (pure, tested)
├── refit.py         redo projection + fusion from stored detections, no VLM
├── vlm.py           Gemini detection
├── crew/            CrewAI agents, tools, session, synonym resolution
├── webapp/          FastAPI viewer + vanilla-JS frontend
│   ├── rooms.py       artifact discovery, PLY reading, decimation
│   ├── refuse.py      re-cluster cached observations (reuses fusion.py)
│   ├── floorplan.py   top-down projection + world<->pixel transform
│   ├── runs.py        subprocess runner, progress from the CLI's own markers
│   └── static/        index.html, app.js, style.css, vendor/three.module.js
├── viewer.py        desktop Open3D view (the CLI's `view` verb)
└── cli.py
```

Two documented artifact interfaces connect the halves:

- **`frames.npz`** — per-frame world-frame pointmap, confidence mask, pose,
  intrinsics. Any reconstructor that can write it (MASt3R-SLAM keyframes, VGGT,
  DUSt3R) feeds the labeling stage unchanged.
- **`observations.json`** — every 2D detection with its box and its lifted 3D
  position, tagged with the object it clustered into. This is what makes the
  viewer able to draw an object back onto the frame it came from, and what makes
  re-fusion free. Its sidecar **`observation_points.npz`** holds the 3D points
  behind each observation, keyed `obs_<index>`; a box can only be fit to points,
  so re-fusion needs them to reproduce what the pipeline produced.

`projection.py` and `fusion.py` contain no LLM and no I/O. That is deliberate:
it is the only way to tell a geometry bug from a VLM bug when output looks wrong.

## Tests

```bash
uv run python -m pytest tests -q
```

186 tests, no API key and no GPU required. `test_pipeline_wiring.py` runs the
whole labeling half against a synthetic room with a stubbed detector, which is
what catches transposed boxes and misaligned masks. `test_webapp.py` drives the
API with FastAPI's `TestClient` against real artifacts on disk.

## Licensing — blocks production

**The reconstruction core is non-commercial.** MASt3R and DUSt3R code *and*
checkpoints are CC BY-NC-SA 4.0. MASt3R's `CHECKPOINTS_NOTICE` additionally
requires agreeing to every training dataset's licence; the mapfree terms are
particularly restrictive.

Everything else — CrewAI, Open3D, PyTorch, OpenCV — is Apache/MIT/BSD.

Production paths:

1. **VGGT-1B-Commercial** — feed-forward, commercially licensed checkpoint,
   emits the same per-frame pointmap + pose structure, so the entire labeling
   half ports unchanged. Strongest option, and the reason `frames.npz` is a
   documented interface.
2. **COLMAP / GLOMAP** — BSD, unrestricted, CPU-capable. No learned priors, so
   more frames and more overlap, and it fails on textureless surfaces.
3. Negotiate a commercial licence with NAVER.

Also: Gemini's **free tier permits Google to train on submitted content.** Fine
for a test room, not for a client site.

## Which way is up

DUSt3R has no gravity vector. Its global aligner anchors the world frame on one
reference camera, and in the OpenCV convention that every model in this stack
uses, a camera's **+Y axis points down the image**. So the raw output is upside
down: `scene.ply` loads inverted in any +Y-up viewer, and "which axis is the
floor" has no answer in the file.

Two independent signals recover it, and `level.py` uses both:

1. **The camera poses.** Column 1 of each camera-to-world pose is that camera's
   own down direction in world coordinates. A handheld phone is roughly upright
   throughout, so the mean of those *is* gravity. This needs no scene structure,
   which matters on a reconstruction too sparse to find a floor in.
2. **The floor plane.** RANSAC over the lowest slice of the cloud, constrained to
   within 35° of the pose estimate so a wall cannot win, then refit on its
   inliers. This corrects the error the poses cannot see: hold the phone tilted
   down for the whole clip and the mean camera down-vector is tilted too.

The reported confidence is the geometric mean of how steadily the phone was
held, how planar the floor came out, and **how closely the two estimates agree** —
so one signal failing drags the score down instead of being averaged away.

Levelling is applied by default at the end of `reconstruct`, and existing rooms
are fixed in place:

```bash
uv run room3d level --room living          # estimate and apply
uv run room3d level --room living --up -y  # or declare it, if the estimate is wrong
uv run room3d reconstruct ... --no-level   # keep the raw aligner frame
```

It is a **rigid transform** — a rotation and a translation, nothing else — so
distances, OBB extents and fusion results are all unchanged. Every 3D artifact
moves together (`frames.npz`, `scene.ply`, `trajectory.txt`, `objects.json`,
`observations.json`), because a half-levelled room, with boxes floating away
from the cloud they describe, is worse than an upside-down one. The applied
matrix is recorded in `level.json`, so it is auditable and exactly invertible,
and re-running is a near no-op rather than a second rotation.

The result is a **canonical frame: +Y up, floor at y = 0, scene centred on the
origin in X/Z.** Yaw is left alone — the minimal rotation onto +Y is used,
because choosing a yaw would mean claiming to know which way the walls run.

Two things fall out of this that were not available before:

- **Object height is now a coordinate.** On `living`, sorting by `centroid[1]`
  puts `chair` at 0.04 m and the wall cabinets at 1.2 m.
- **Scale became checkable.** The cameras sit 0.68–0.81 m above the floor and the
  ceiling is at 1.37 m. A phone is held at about 1.4 m and ceilings are about
  2.4 m, so this reconstruction is roughly **1.8× under-scaled** — a Gate 3
  finding that was not visible while the room had no floor.

Why the floor plan needed this: a `PlanTransform` is axis-aligned, so it can only
render a room whose up direction *is* a coordinate axis. On an unlevelled room
the best it can do is snap to the nearest axis, so the viewer scales its
confidence by how far that snap had to reach and tells you to run `room3d level`.
Before levelling, `living` scored **0.07** and the tab warned the plan might be a
side view; after, it scores **0.82** and the plan shows the sofa from above.

## How the 3D boxes are fit

Levelling is what makes a good box possible, so this section follows that one.

A detection's pixels index straight into the frame's world-frame pointmap, which
gives a cloud per observation. The question is what box to wrap around it, and
the obvious answer — PCA — is wrong here. **You only ever see one side of a
sofa.** PCA on the front surface finds the axes of that slab, not of the sofa,
and returns a box tilted at whatever angle the visible face happened to make.

`fit_gravity_aligned_box` replaces the guess with a fact: furniture stands on a
floor, so the vertical axis is gravity and only the **yaw** is unknown. It sweeps
yaw at 1° and keeps the angle with the smallest ground footprint — rotating
calipers with a fixed step, which is all the accuracy these clouds support.
Extents come from the 1st/99th percentile rather than min/max, so the handful of
pixels that leaked onto the wall behind the object cannot stretch the box across
the room. Finally `snap_to_floor` pulls an underside already within 15 cm of
`y = 0` onto it, which recovers the depth every reconstruction is missing under
a sofa. A shelf on a wall is nowhere near the floor and is left alone.

Fusion then fits **one** box to the pooled points of every observation in a
cluster. That is the only honest way to build it: per-frame boxes express their
extents along *different* axes, so combining two of them element-wise describes
no box at all. This is why `Observation` carries its points (capped at 2048 per
frame, 20 000 pooled) and why they are persisted to `observation_points.npz`.

Two invariants are worth stating because violating either produces a box that
looks plausible in isolation and is nonsense against the scene:

- `points_inside_fraction(obb, points)` is the coherence check. A box built from
  a centre, an extent and a rotation taken from different sources can contain
  almost none of the geometry it claims to describe.
- Extents, rotation and centre must come from **one** fit. Never mix.

Clustering uses **nearest-member** linkage, not distance to the cluster mean. A
sofa filmed by walking past it produces views that each overlap their neighbours
and not the far end; once two merge, their mean sits between them and every later
view is measured from a place no observation ever was, so the far half splits off
as a second sofa.

## Phone video rotation

Phones shoot portrait but store frames landscape with a `rotate` flag. OpenCV
returns the raw unrotated frames, so without handling this every frame arrives
**on its side** — which costs little geometrically, but a great deal at the VLM,
which is being asked to name furniture in a picture lying sideways.

`extract_frames` reads `CAP_PROP_ORIENTATION_META` and rotates before anything
else sees the pixels, printing `[extract] applied 90deg rotation from video
metadata` when it fires. The bundled sample measures 90°.

## Known limits

- Large blank walls, glass and mirrors produce holes — a DUSt3R-family weakness.
- No loop closure; drift over a long loop is uncorrected.
- Metric scale comes from the `_metric` checkpoint but global alignment
  optimises per-pair scale factors, so it can drift. Verify against a tape
  measure; use `--scale-ref` if it has.
- Object extents are approximate, and systematically *under*-estimated in depth:
  only the visible surface is ever reconstructed, so a sofa's box reaches to its
  front face and not to the wall behind it. Floor snapping recovers the vertical
  direction; nothing yet recovers the unseen depth. Category size priors or
  symmetry completion would, and neither is implemented.
- Over-segmentation is the dominant remaining error, not box shape: on the
  sample living room 42 of 50 objects are single sightings. That is a detection
  and clustering problem, above the geometry described here.
