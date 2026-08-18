"""The level-aware box fit has to survive the trip through the real pipeline.

A correct `fit_gravity_aligned_box` is worth nothing if the tools never hand it
an up vector, or if the points it needs are dropped between projection and
fusion. These tests run the actual tools over a synthetic levelled room.
"""

import json

import numpy as np
import pytest

from room3d.artifacts import Reconstruction, save_observation_points
from room3d.config import LabelConfig
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.session import LabelingSession
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from room3d.level import UP_VECTOR
from room3d.projection import points_inside_fraction
from room3d.vlm import Detection
from room3d.webapp.refuse import load_observations, refuse

H, W, N = 48, 64, 4

# A 2.0 x 0.8 x 0.9 sofa standing on the floor (y = 0), yawed 25 degrees.
SOFA_SIZE = np.array([2.0, 0.8, 0.9])
SOFA_CENTRE = np.array([0.0, 0.4, 2.0])
SOFA_YAW = 25.0


def sofa_surface(n_u: int, n_v: int, u_span: tuple[float, float]) -> np.ndarray:
    """A patch of the sofa's front face, as an (n_u, n_v, 3) pixel grid."""
    u = np.linspace(*u_span, n_v)                     # along its length
    v = np.linspace(-0.5, 0.5, n_u)                   # up its height
    gu, gv = np.meshgrid(u, v)

    local = np.stack(
        [gu * SOFA_SIZE[0], gv * SOFA_SIZE[1], np.full_like(gu, 0.45)], axis=-1
    )
    t = np.radians(SOFA_YAW)
    R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    return local @ R.T + SOFA_CENTRE


def build_recon() -> Reconstruction:
    """Four cameras, each seeing an overlapping slice of the sofa in a real room.

    The room needs a genuine floor, not just a backdrop: levelling estimates up
    from the floor plane, and a scene without one is a scene where the pipeline
    correctly refuses to align anything.
    """
    pts3d = np.zeros((N, H, W, 3), dtype=np.float32)

    # Upper rows: the wall behind everything. Lower rows: the floor at y = 0.
    pts3d[:, :36, :, 1] = 2.5
    pts3d[:, :36, :, 2] = 8.0
    floor_x = np.linspace(-3.0, 3.0, W, dtype=np.float32)
    floor_z = np.linspace(5.0, 1.0, H - 36, dtype=np.float32)
    pts3d[:, 36:, :, 0] = floor_x[None, None, :]
    pts3d[:, 36:, :, 1] = 0.0
    pts3d[:, 36:, :, 2] = floor_z[None, :, None]

    spans = [(-0.50, -0.05), (-0.30, 0.15), (-0.10, 0.35), (0.05, 0.50)]
    for i, span in enumerate(spans):
        pts3d[i, 12:36, 20:44] = sofa_surface(24, 24, span)

    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, 0, 3] = np.linspace(-1.0, 1.0, N)
    # A proper (det = +1) 180-degree rotation about the world Z axis: local +Y
    # (down the image) maps to world -Y, i.e. world +Y is up; local +Z (forward)
    # maps to world +Z, which is where this room's geometry actually lives (z in
    # 1..8). Flipping only Y, as an earlier version of this fixture did, points
    # the camera's forward axis at world -Z instead -- behind every camera, so a
    # frame cannot even see the points it supposedly recorded. That was invisible
    # to every test above, none of which reprojects through the camera model; it
    # surfaces as soon as something does (room3d.camera.visible_in_frame, used by
    # consensus.py's cross-view agreement).
    poses[:, 0, 0] = -1.0
    poses[:, 1, 1] = -1.0                              # camera +Y is down, world +Y up

    # A short focal length, not the tighter 50 that was here before real camera
    # reprojection was ever exercised against this room. Each frame's box is a
    # single fixed pixel window (`sofa_detection` is reused verbatim for every
    # frame, the way a box-fit-only fixture can afford to), not one that tracks
    # the sofa's true per-frame parallax the way a real detector's boxes would.
    # A wide field of view keeps that fixed window forgiving enough of the
    # resulting few-pixel shifts between frames for cross-view agreement to still
    # recognise the four partial views as one object; box-fit itself never reads
    # these intrinsics; only Task 8's consensus voting does.
    intrinsics = np.tile(
        np.array([[20, 0, W / 2], [0, 20, H / 2], [0, 0, 1]], dtype=np.float32), (N, 1, 1)
    )
    return Reconstruction(
        images=np.full((N, H, W, 3), 128, dtype=np.uint8),
        pts3d=pts3d,
        conf_mask=np.ones((N, H, W), dtype=bool),
        poses=poses,
        intrinsics=intrinsics,
        frame_ids=np.arange(N, dtype=np.int32),
    )


def sofa_detection() -> Detection:
    """The pixel block holding the sofa, in Gemini's 0-1000 y-first convention."""
    return Detection(
        label="sofa",
        box_2d=[
            int(12 / H * 1000), int(20 / W * 1000),
            int(36 / H * 1000), int(44 / W * 1000),
        ],
        confidence=0.9,
    )


def labelled_session(**overrides) -> LabelingSession:
    config = LabelConfig(min_mask_points=10, **overrides)
    session = LabelingSession(recon=build_recon(), config=config)
    session.selected_frames = list(range(N))
    session.detections = {i: [sofa_detection()] for i in range(N)}
    return session


def run_tools(session: LabelingSession) -> None:
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)


# --- the session knows which way is up ---------------------------------------


def test_session_recovers_the_up_vector_from_the_reconstruction():
    up, _ = labelled_session().gravity
    assert up is not None
    assert float(up @ UP_VECTOR) > 0.95


def test_gravity_can_be_switched_off():
    up, floor = labelled_session(gravity_aligned_boxes=False).gravity
    assert up is None and floor is None


# --- projection uses it ------------------------------------------------------


def test_projected_observations_are_level_and_keep_their_points():
    session = labelled_session()
    ProjectDetectionsTool(session)._run()

    assert len(session.observations) == N
    for obs in session.observations:
        assert np.allclose(obs.obb.R[:, 1], session.gravity[0], atol=1e-9)
        assert obs.points is not None and len(obs.points) > 0


def test_retained_points_respect_the_configured_cap():
    session = labelled_session(max_points_per_observation=25)
    ProjectDetectionsTool(session)._run()
    assert all(len(o.points) <= 25 for o in session.observations)


# --- fusion produces one coherent box ----------------------------------------


def test_the_fused_sofa_box_is_level_and_contains_every_observation():
    session = labelled_session()
    run_tools(session)

    assert len(session.objects) == 1
    obb = session.objects[0].obb
    assert np.allclose(obb.R[:, 1], UP_VECTOR, atol=1e-2)

    pooled = np.vstack([o.points for o in session.observations])
    assert points_inside_fraction(obb, pooled) > 0.95


def test_the_fused_box_spans_more_than_any_single_frame_saw():
    session = labelled_session()
    run_tools(session)

    fused = max(session.objects[0].obb.extent)
    widest_frame = max(max(o.obb.extent) for o in session.observations)
    assert fused > widest_frame + 0.15


def test_the_fused_box_recovers_the_true_sofa_length():
    session = labelled_session()
    run_tools(session)
    assert max(session.objects[0].obb.extent) == pytest.approx(2.0, abs=0.2)


# --- persistence keeps re-fusion honest --------------------------------------


def test_points_round_trip_through_disk_so_refusion_matches_the_pipeline(tmp_path):
    session = labelled_session()
    run_tools(session)

    obs_path = tmp_path / "observations.json"
    obs_path.write_text(json.dumps(build_observations_doc(session, "room")))
    save_observation_points(obs_path, session.observations)

    reloaded = load_observations(obs_path)
    assert all(o.points is not None for o in reloaded.observations)

    objects = refuse(reloaded, up=UP_VECTOR, floor_height=0.0)
    assert len(objects) == 1
    assert np.allclose(objects[0].obb.extent, session.objects[0].obb.extent, atol=1e-6)


def test_refusion_still_works_when_no_points_file_exists(tmp_path):
    session = labelled_session()
    run_tools(session)

    obs_path = tmp_path / "observations.json"
    obs_path.write_text(json.dumps(build_observations_doc(session, "room")))

    reloaded = load_observations(obs_path)
    assert all(o.points is None for o in reloaded.observations)
    assert len(refuse(reloaded, up=UP_VECTOR)) == 1
