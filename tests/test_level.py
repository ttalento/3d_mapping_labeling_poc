"""Standing the reconstruction upright.

The bug these tests exist for: DUSt3R anchors the world frame on a camera, and
in the OpenCV convention a camera's +Y points *down* the image. So the raw scene
is upside down, and the old "the vertical axis is the one with the smallest
extent" heuristic picked the wrong axis entirely on a real room.

Every test builds a synthetic room with a *known* gravity direction and checks
we recover it, because the only way to tell a good estimate from a lucky one is
to know the answer in advance.
"""

import json

import numpy as np
import pytest

from room3d import level as L
from room3d.artifacts import Reconstruction, load_frames_npz, save_frames_npz


# --- fixtures ---------------------------------------------------------------


def rotation(axis, degrees):
    """Rodrigues rotation, so tests can tilt a room by a known amount."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    t = np.radians(degrees)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def make_room(up=(0.0, 1.0, 0.0), floor_height=0.0, n=8000, seed=0):
    """A 4x3 m floor slab plus two walls, with `up` as the true vertical."""
    rng = np.random.default_rng(seed)
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)

    # Build in a canonical Y-up frame, then rotate onto the requested up.
    floor = np.stack(
        [rng.uniform(-2, 2, n), rng.normal(0, 0.005, n), rng.uniform(-1.5, 1.5, n)], axis=1
    )
    wall_a = np.stack(
        [np.full(n // 2, -2.0), rng.uniform(0, 2.4, n // 2), rng.uniform(-1.5, 1.5, n // 2)],
        axis=1,
    )
    wall_b = np.stack(
        [rng.uniform(-2, 2, n // 2), rng.uniform(0, 2.4, n // 2), np.full(n // 2, 1.5)], axis=1
    )
    pts = np.concatenate([floor, wall_a, wall_b])

    R = L.rotation_between(np.array([0.0, 1.0, 0.0]), up)
    return pts @ R.T + up * floor_height


def make_poses(up=(0.0, 1.0, 0.0), n=6, jitter_deg=0.0, seed=0):
    """Camera-to-world poses for a phone held upright, i.e. camera +Y ~ -up."""
    rng = np.random.default_rng(seed)
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)

    # Camera axes: +Y down (= -up), +Z forward (horizontal), +X = Y x Z.
    forward = np.array([0.0, 0.0, 1.0])
    forward = forward - (forward @ up) * up
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ up) * up
    forward /= np.linalg.norm(forward)

    down = -up
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)

    base = np.stack([right, down, forward], axis=1)     # columns are the axes

    poses = np.zeros((n, 4, 4))
    poses[:, 3, 3] = 1.0
    for i in range(n):
        R = base
        if jitter_deg:
            wobble = rotation(rng.normal(size=3), rng.normal(0, jitter_deg))
            R = wobble @ base
        poses[i, :3, :3] = R
        poses[i, :3, 3] = up * 1.4 + np.array([0.3 * i, 0.0, 0.0])
    return poses


# --- the pose estimator -----------------------------------------------------


def test_up_from_poses_recovers_gravity_from_an_upright_phone():
    up = np.array([0.0, 1.0, 0.0])
    got, tilt = L.up_from_poses(make_poses(up))
    assert np.allclose(got, up, atol=1e-6)
    assert tilt == pytest.approx(0.0, abs=1e-6)


def test_up_from_poses_survives_a_wobbly_hand():
    up = np.array([0.2, 0.9, -0.3])
    up /= np.linalg.norm(up)
    got, tilt = L.up_from_poses(make_poses(up, n=12, jitter_deg=8.0, seed=3))
    assert np.degrees(np.arccos(np.clip(got @ up, -1, 1))) < 5.0
    assert 0.0 < tilt < 25.0


def test_up_from_poses_handles_the_real_camera_convention():
    """A DUSt3R world frame is a camera frame, so raw up is -Y, not +Y.

    This is the actual bug: with the reference camera at identity, "world" has
    +Y pointing down the image, and any viewer assuming +Y up renders the room
    upside down.
    """
    poses = np.tile(np.eye(4), (4, 1, 1))
    up, _ = L.up_from_poses(poses)
    assert np.allclose(up, [0.0, -1.0, 0.0])


# --- the floor fit ----------------------------------------------------------


def test_fit_floor_plane_finds_the_floor_not_a_wall():
    up = np.array([0.0, 1.0, 0.0])
    fit = L.fit_floor_plane(make_room(up), up)
    assert fit is not None
    normal, offset, inliers = fit
    assert np.degrees(np.arccos(np.clip(normal @ up, -1, 1))) < 2.0
    assert offset == pytest.approx(0.0, abs=0.02)
    assert inliers > 0.5


def test_fit_floor_plane_corrects_a_tilted_pose_prior():
    """The point of the floor fit: poses alone cannot see a consistent tilt."""
    up = np.array([0.0, 1.0, 0.0])
    tilted = rotation([1.0, 0.0, 0.0], 12.0) @ up          # a 12 deg wrong prior

    normal, _, _ = L.fit_floor_plane(make_room(up), tilted)
    assert np.degrees(np.arccos(np.clip(normal @ up, -1, 1))) < 3.0


def test_fit_floor_plane_returns_none_when_there_is_no_floor():
    rng = np.random.default_rng(0)
    blob = rng.normal(size=(4000, 3))
    assert L.fit_floor_plane(blob, [0.0, 1.0, 0.0], tolerance=0.005) is None


def test_fit_floor_plane_rejects_a_plane_beyond_the_tilt_limit():
    """A wall is a huge plane; it must not be allowed to win on inlier count."""
    up = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(1)
    # A cloud that is mostly one vertical wall, with a thin floor.
    wall = np.stack(
        [np.full(9000, 0.0) + rng.normal(0, 0.004, 9000),
         rng.uniform(0, 2.4, 9000), rng.uniform(-2, 2, 9000)], axis=1
    )
    floor = np.stack(
        [rng.uniform(-2, 2, 1200), rng.normal(0, 0.004, 1200), rng.uniform(-2, 2, 1200)],
        axis=1,
    )
    fit = L.fit_floor_plane(np.concatenate([wall, floor]), up)
    if fit is not None:
        normal, _, _ = fit
        assert abs(normal @ up) > np.cos(np.radians(35))


# --- the combined estimate --------------------------------------------------


def test_estimate_up_beats_the_extent_heuristic_on_a_tall_narrow_room():
    """The failure that motivated all of this.

    A partial capture can easily be *taller* than it is deep, and then "the
    smallest extent is vertical" picks a horizontal axis with confidence.
    """
    rng = np.random.default_rng(0)
    up = np.array([0.0, 1.0, 0.0])
    # 4 m wide, 3 m tall, only 0.8 m deep -- depth is the smallest extent.
    pts = np.stack(
        [rng.uniform(-2, 2, 9000), rng.uniform(0, 3, 9000), rng.uniform(-0.4, 0.4, 9000)],
        axis=1,
    )
    floor = np.stack(
        [rng.uniform(-2, 2, 4000), rng.normal(0, 0.005, 4000), rng.uniform(-0.4, 0.4, 4000)],
        axis=1,
    )
    pts = np.concatenate([pts, floor])

    assert L._up_from_extent(pts).up[2] != 0.0            # the old answer: wrong axis

    est = L.estimate_up(pts, make_poses(up))
    assert np.degrees(np.arccos(np.clip(est.up @ up, -1, 1))) < 5.0


def test_estimate_up_scores_agreement_between_the_two_signals():
    up = np.array([0.0, 1.0, 0.0])
    est = L.estimate_up(make_room(up), make_poses(up))
    assert est.source == "poses+floor"
    assert est.confidence > 0.8
    assert est.agreement_deg < 5.0


def test_estimate_up_falls_back_to_poses_when_no_floor_is_visible():
    rng = np.random.default_rng(0)
    up = np.array([0.0, 1.0, 0.0])
    blob = rng.normal(size=(4000, 3))
    est = L.estimate_up(blob, make_poses(up), tolerance=0.001)
    assert est.source == "poses"
    assert np.allclose(est.up, up, atol=1e-6)
    # Unverified by scene structure, so it must not claim full confidence.
    assert est.confidence < 0.75


def test_estimate_up_without_poses_says_it_is_guessing():
    est = L.estimate_up(make_room())
    assert est.source == "extent"
    assert est.confidence <= 0.5
    assert est.notes


# --- the transform ----------------------------------------------------------


def test_rotation_between_handles_the_antiparallel_case():
    """up = -target is exactly the upside-down case, and cross(a, b) is zero."""
    R = L.rotation_between([0.0, -1.0, 0.0], [0.0, 1.0, 0.0])
    assert np.allclose(R @ [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_levelling_puts_up_on_plus_y_and_the_floor_at_zero():
    up = np.array([0.1, -0.95, 0.3])           # upside down and tilted
    up /= np.linalg.norm(up)
    pts = make_room(up, floor_height=2.7)

    est = L.estimate_up(pts, make_poses(up))
    T = L.levelling_transform(est, pts)
    out = L.transform_points(T, pts)

    # The floor slab is the dense low sheet; it should now sit on y = 0.
    floor_y = np.quantile(out[:, 1], 0.02)
    assert floor_y == pytest.approx(0.0, abs=0.05)
    assert out[:, 1].max() > 2.0               # walls go up, not down
    assert np.median(out[:, 0]) == pytest.approx(0.0, abs=0.05)


def test_levelling_is_rigid_so_no_geometry_changes():
    """The whole safety argument for doing this to saved artifacts."""
    rng = np.random.default_rng(0)
    up = np.array([0.2, -0.9, 0.35])
    pts = make_room(up)
    T = L.levelling_transform(L.estimate_up(pts, make_poses(up)), pts)

    a, b = rng.choice(len(pts), 400), rng.choice(len(pts), 400)
    before = np.linalg.norm(pts[a] - pts[b], axis=1)
    after = np.linalg.norm(L.transform_points(T, pts)[a] - L.transform_points(T, pts)[b], axis=1)
    assert np.allclose(before, after, atol=1e-9)


def test_levelling_an_already_level_room_is_a_near_no_op():
    """Guards against `room3d level` double-rotating a room when re-run."""
    up = np.array([0.0, 1.0, 0.0])
    pts = make_room(up)
    T = L.levelling_transform(L.estimate_up(pts, make_poses(up)), pts)
    assert np.allclose(T[:3, :3], np.eye(3), atol=0.02)


def test_transform_poses_moves_cameras_with_the_scene():
    up = np.array([0.0, -1.0, 0.0])
    poses = make_poses(up)
    T = L.levelling_transform(L.estimate_up(make_room(up), poses), make_room(up))

    moved = L.transform_poses(T, poses)
    new_up, _ = L.up_from_poses(moved)
    # Not exact: the estimate is refined by a RANSAC floor fit, so it lands
    # within a hundredth of a degree of vertical rather than on it.
    assert np.allclose(new_up, [0.0, 1.0, 0.0], atol=1e-3)
    # Still valid rotations, not squashed by a sloppy multiply.
    for R in moved[:, :3, :3]:
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_transform_records_moves_centroids_and_box_rotations():
    T = np.eye(4)
    T[:3, :3] = L.rotation_between([0.0, -1.0, 0.0], [0.0, 1.0, 0.0])
    T[:3, 3] = [1.0, 2.0, 3.0]

    rec = {
        "id": "obj_001",
        "centroid": [1.0, -1.0, 0.0],
        "obb": {"center": [1.0, -1.0, 0.0], "extent": [0.5, 0.6, 0.7],
                "R": np.eye(3).tolist()},
    }
    out = L.transform_records(T, [rec])[0]

    assert out["obb"]["extent"] == [0.5, 0.6, 0.7]          # extents are invariant
    assert np.allclose(out["centroid"], T[:3, :3] @ rec["centroid"] + T[:3, 3], atol=1e-4)
    assert np.allclose(out["obb"]["R"], T[:3, :3])
    assert rec["centroid"] == [1.0, -1.0, 0.0]              # input not mutated


def test_forced_estimate_accepts_signed_axes():
    pts = make_room()
    assert np.allclose(L._forced_estimate("-y", pts).up, [0.0, -1.0, 0.0])
    assert np.allclose(L._forced_estimate("Z", pts).up, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        L._forced_estimate("up", pts)


# --- the whole room on disk -------------------------------------------------


@pytest.fixture
def upside_down_room(tmp_path):
    """A room stored the way the aligner actually leaves one: +Y is down."""
    up = np.array([0.0, -1.0, 0.0])
    n_frames, h, w = 4, 24, 32

    pts = make_room(up, n=n_frames * h * w // 2)
    pts = np.resize(pts, (n_frames, h, w, 3)).astype(np.float32)

    recon = Reconstruction(
        images=np.full((n_frames, h, w, 3), 128, dtype=np.uint8),
        pts3d=pts,
        conf_mask=np.ones((n_frames, h, w), dtype=bool),
        poses=make_poses(up, n=n_frames).astype(np.float32),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        frame_ids=np.arange(n_frames, dtype=np.int32),
    )
    save_frames_npz(tmp_path / "frames.npz", recon)

    (tmp_path / "objects.json").write_text(json.dumps({
        "room": "demo",
        "objects": [{
            "id": "obj_001", "label": "sofa", "centroid": [0.5, -0.4, 0.2],
            "obb": {"center": [0.5, -0.4, 0.2], "extent": [1.8, 0.8, 0.9],
                    "R": np.eye(3).tolist()},
        }],
    }))
    (tmp_path / "observations.json").write_text(json.dumps({
        "room": "demo",
        "observations": [{
            "id": 0, "object_id": "obj_001", "frame_idx": 0, "label": "sofa",
            "centroid": [0.5, -0.4, 0.2],
            "obb": {"center": [0.5, -0.4, 0.2], "extent": [1.8, 0.8, 0.9],
                    "R": np.eye(3).tolist()},
        }],
    }))
    return tmp_path


def test_level_room_rewrites_every_artifact_together(upside_down_room):
    """A half-levelled room is worse than an upside-down one: boxes would
    float away from the cloud they describe."""
    result = L.level_room(upside_down_room, verbose=False)
    assert result.rotation_deg > 90.0

    recon = load_frames_npz(upside_down_room / "frames.npz")
    up, _ = L.up_from_poses(recon.poses)
    assert np.allclose(up, [0.0, 1.0, 0.0], atol=1e-3)

    cloud = recon.pts3d[recon.conf_mask]
    assert np.quantile(cloud[:, 1], 0.02) == pytest.approx(0.0, abs=0.1)

    obj = json.loads((upside_down_room / "objects.json").read_text())["objects"][0]
    expected = L.transform_points(result.transform, np.array([0.5, -0.4, 0.2]))
    assert np.allclose(obj["centroid"], expected, atol=1e-3)
    assert obj["obb"]["extent"] == [1.8, 0.8, 0.9]

    obs = json.loads((upside_down_room / "observations.json").read_text())["observations"][0]
    assert np.allclose(obs["centroid"], expected, atol=1e-3)

    assert (upside_down_room / "scene.ply").exists()
    assert (upside_down_room / "trajectory.txt").exists()


def test_level_room_records_an_invertible_transform(upside_down_room):
    result = L.level_room(upside_down_room, verbose=False)
    record = L.load_level_record(upside_down_room)

    assert record["convention"] == "y_up_floor_at_zero"
    T = np.asarray(record["applied"])
    assert np.allclose(T, result.transform)

    original = np.array([0.5, -0.4, 0.2])
    moved = L.transform_points(T, original)
    assert np.allclose(L.transform_points(np.linalg.inv(T), moved), original, atol=1e-6)


def test_level_room_twice_does_not_double_rotate(upside_down_room):
    L.level_room(upside_down_room, verbose=False)
    first = load_frames_npz(upside_down_room / "frames.npz").pts3d.copy()

    second_result = L.level_room(upside_down_room, verbose=False)
    second = load_frames_npz(upside_down_room / "frames.npz").pts3d

    assert second_result.rotation_deg < 2.0
    assert np.allclose(first, second, atol=0.05)


def test_level_room_honours_a_manual_axis(upside_down_room):
    L.level_room(upside_down_room, up="-y", verbose=False)
    record = L.load_level_record(upside_down_room)
    assert record["estimate"]["source"] == "manual"
    assert record["rotation_deg"] == pytest.approx(180.0, abs=1.0)


def test_level_room_refuses_a_room_with_no_reconstruction(tmp_path):
    with pytest.raises(FileNotFoundError):
        L.level_room(tmp_path, verbose=False)
