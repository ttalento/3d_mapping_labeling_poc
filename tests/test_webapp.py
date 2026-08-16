"""Viewer backend: artifact loading, re-fusion, floor-plan geometry, endpoints.

The re-fusion tests matter most. Tuning sliders that quietly disagree with what
the pipeline does would be worse than no sliders at all, so the reproduction
test pins re-fusion to the same code path.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from room3d.artifacts import save_ply, save_trajectory_tum
from room3d.webapp import floorplan as fp
from room3d.webapp import rooms as R
from room3d.webapp.refuse import load_observations, objects_doc, refuse, save_objects
from room3d.webapp.server import create_app

# --- fixtures ---------------------------------------------------------------


def make_observation(i, frame, label, centroid, extent=(0.4, 0.4, 0.4)):
    return {
        "id": i,
        "object_id": None,
        "frame_idx": frame,
        "label": label,
        "vlm_confidence": 0.9,
        "centroid": list(centroid),
        "obb": {"center": list(centroid), "extent": list(extent),
                "R": np.eye(3).tolist()},
        "n_points": 500,
        "support": 0.8,
        "box_px": [10, 20, 60, 80],
    }


@pytest.fixture
def room_dir(tmp_path):
    """A complete room: cloud, trajectory, objects, observations, frames."""
    root = tmp_path / "out" / "demo"
    (root / "frames").mkdir(parents=True)

    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, 0, -2], [2, 2.4, 2], size=(4000, 3)).astype(np.float32)
    cols = rng.integers(0, 256, (4000, 3), dtype=np.uint8)
    save_ply(root / "scene.ply", pts, cols)

    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 0, 3] = [0.0, 0.5, 1.0, 1.5]
    save_trajectory_tum(root / "trajectory.txt", poses)

    observations = [
        make_observation(0, 0, "chair", (0.0, 0.5, 0.0)),
        make_observation(1, 1, "chair", (0.05, 0.5, 0.02)),
        make_observation(2, 2, "chair", (3.0, 0.5, 0.0)),
        make_observation(3, 3, "lamp", (0.0, 1.5, 0.0)),
    ]
    (root / "observations.json").write_text(json.dumps({
        "room": "demo", "image_hw": [512, 384], "frames_labeled": [0, 1, 2, 3],
        "observations": observations,
    }))

    (root / "objects.json").write_text(json.dumps({
        "room": "demo", "units": "meters", "scale_verified": False,
        "n_frames_labeled": 4, "objects": [],
    }))

    import cv2
    for i in range(4):
        cv2.imwrite(str(root / "frames" / f"{i:03d}.png"),
                    rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))

    return root


@pytest.fixture
def client(room_dir):
    return TestClient(create_app(out_dir=room_dir.parent, project_root=room_dir.parents[1]))


# --- ply / cloud ------------------------------------------------------------


def test_read_ply_round_trips(tmp_path):
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    cols = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    path = tmp_path / "c.ply"
    save_ply(path, pts, cols)

    got_pts, got_cols = R.read_ply(path)
    assert np.allclose(got_pts, pts)
    assert np.array_equal(got_cols, cols)


def test_decimate_returns_exactly_max_points():
    pts = np.random.default_rng(0).normal(size=(5000, 3)).astype(np.float32)
    cols = np.zeros((5000, 3), np.uint8)
    out_pts, out_cols = R.decimate(pts, cols, 1000)
    assert len(out_pts) == len(out_cols) == 1000


def test_decimate_is_a_noop_below_the_cap():
    pts = np.zeros((10, 3), np.float32)
    cols = np.zeros((10, 3), np.uint8)
    assert R.decimate(pts, cols, 100)[0].shape == (10, 3)


def test_decimate_preserves_the_bounding_box_roughly():
    """A subsample that lost a whole side of the room would misframe the view."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(200_000, 3)).astype(np.float32)
    cols = np.zeros((200_000, 3), np.uint8)

    small, _ = R.decimate(pts, cols, 20_000)
    assert np.allclose(small.min(axis=0), pts.min(axis=0), atol=0.15)
    assert np.allclose(small.max(axis=0), pts.max(axis=0), atol=0.15)


def test_decimate_is_deterministic():
    pts = np.random.default_rng(1).normal(size=(5000, 3)).astype(np.float32)
    cols = np.zeros((5000, 3), np.uint8)
    a, _ = R.decimate(pts, cols, 500)
    b, _ = R.decimate(pts, cols, 500)
    assert np.array_equal(a, b), "view would shimmer between reloads"


def test_pack_cloud_framing_is_readable():
    import struct
    pts = np.array([[1, 2, 3]], dtype=np.float32)
    cols = np.array([[7, 8, 9]], dtype=np.uint8)
    blob = R.pack_cloud(pts, cols)

    assert struct.unpack("<I", blob[:4])[0] == 1
    assert np.allclose(np.frombuffer(blob[4:16], dtype="<f4"), [1, 2, 3])
    assert np.array_equal(np.frombuffer(blob[16:19], dtype=np.uint8), [7, 8, 9])


# --- trajectory -------------------------------------------------------------


def test_trajectory_steps_and_span(room_dir):
    poses = R.read_trajectory(room_dir / "trajectory.txt")
    assert [p["step"] for p in poses] == pytest.approx([0.0, 0.5, 0.5, 0.5])

    stats = R.trajectory_stats(poses)
    assert stats["span_max"] == pytest.approx(1.5)
    assert stats["path_length"] == pytest.approx(1.5)


def test_trajectory_stats_on_empty_input():
    assert R.trajectory_stats([])["n_poses"] == 0


# --- re-fusion --------------------------------------------------------------


def test_refuse_merges_near_and_splits_far(room_dir):
    obs = load_observations(room_dir / "observations.json")
    objects = refuse(obs)

    labels = sorted(o.label for o in objects)
    assert labels == ["chair", "chair", "lamp"], labels
    biggest = max(objects, key=lambda o: o.n_observations)
    assert biggest.n_observations == 2


def test_refuse_reproduces_the_pipeline_clustering(room_dir):
    """Re-fusion must be the same code path, not a lookalike."""
    from room3d.fusion import cluster_observations

    obs = load_observations(room_dir / "observations.json")
    direct = cluster_observations(obs.observations)
    viaapi = refuse(obs)

    assert [o.label for o in direct] == [o.label for o in viaapi]
    assert [o.n_observations for o in direct] == [o.n_observations for o in viaapi]


def test_larger_radius_never_increases_object_count(room_dir):
    obs = load_observations(room_dir / "observations.json")
    counts = [len(refuse(obs, radius_floor=r)) for r in (0.1, 0.5, 1.0, 2.0, 4.0)]
    assert counts == sorted(counts, reverse=True), counts


@pytest.fixture
def spread_obs(tmp_path):
    """Same-label observations at a ladder of separations.

    A fixture whose objects all sit far apart cannot detect a control that does
    nothing, because nothing is what should happen. The separations here
    straddle the slider's range so each setting has a different right answer.
    """
    path = tmp_path / "observations.json"
    xs = [0.0, 0.15, 0.45, 0.9, 1.8, 3.6]
    path.write_text(json.dumps({
        "room": "spread", "image_hw": [512, 384], "frames_labeled": list(range(len(xs))),
        "observations": [make_observation(i, i, "box", (x, 0.0, 0.0), extent=(0.3, 0.3, 0.3))
                         for i, x in enumerate(xs)],
    }))
    return load_observations(path)


def test_the_radius_slider_actually_changes_something(spread_obs):
    """Monotonicity alone is satisfied by a control that does nothing.

    This caught a real problem: over the slider's original 0.05-2.0 range the
    object count for the `living` room never moved, because the merge radius is
    max(floor, scale x mean diagonal) and the size term already dominated. The
    fix was to widen the range and show the effective radius.
    """
    counts = {len(refuse(spread_obs, radius_floor=r)) for r in (0.05, 1.0, 5.0)}
    assert len(counts) > 1, f"radius_floor had no effect across its range: {counts}"


def test_radius_scale_also_changes_something(spread_obs):
    counts = {len(refuse(spread_obs, radius_scale=s, radius_floor=0.05))
              for s in (0.0, 2.0, 8.0)}
    assert len(counts) > 1, f"radius_scale had no effect across its range: {counts}"


def test_widest_radius_collapses_everything(spread_obs):
    assert len(refuse(spread_obs, radius_floor=50.0)) == 1


def test_radius_summary_reports_which_term_wins(room_dir):
    from room3d.webapp.refuse import radius_summary

    obs = load_observations(room_dir / "observations.json")

    by_floor = radius_summary(obs, radius_floor=10.0, radius_scale=0.5)
    assert by_floor["driven_by"] == "floor"
    assert by_floor["effective_radius"] == pytest.approx(10.0)

    by_size = radius_summary(obs, radius_floor=0.01, radius_scale=2.0)
    assert by_size["driven_by"] == "size"
    assert by_size["merge_distance_no_overlap"] == pytest.approx(
        by_size["effective_radius"] / 2, rel=1e-3)


def test_min_observations_filters_singletons(room_dir):
    obs = load_observations(room_dir / "observations.json")
    assert all(o.n_observations >= 2 for o in refuse(obs, min_observations=2))


def test_observations_round_trip_preserves_boxes(room_dir):
    obs = load_observations(room_dir / "observations.json")
    assert obs.n == 4
    assert obs.image_hw == (512, 384)
    assert all(o.box_px == (10, 20, 60, 80) for o in obs.observations)


def test_object_records_point_back_at_their_observations(room_dir):
    obs = load_observations(room_dir / "observations.json")
    objects = refuse(obs)

    all_ids = sorted(i for o in objects for i in o.observation_ids)
    assert all_ids == [0, 1, 2, 3], "every observation must belong to exactly one object"


def test_save_objects_keeps_a_backup(room_dir):
    path = room_dir / "objects.json"
    original = path.read_text()
    save_objects(path, objects_doc("demo", []))

    assert (room_dir / "objects.prev.json").read_text() == original
    assert json.loads(path.read_text())["objects"] == []


# --- floor plan -------------------------------------------------------------


def test_estimate_up_axis_finds_the_flat_direction():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-3, -0.2, -3], [3, 0.2, 3], size=(5000, 3))
    axis, _, confidence = fp.estimate_up_axis(pts)
    assert axis == 1
    # Points alone can only fall back to the shape heuristic, which is weak
    # enough that it picked the wrong axis on the first real room. It gets the
    # right answer here, and still must not sound sure of itself.
    assert 0.1 < confidence < 0.6


def test_estimate_up_axis_is_confident_when_given_camera_poses():
    """Poses are what turn the guess into a measurement."""
    from test_level import make_poses, make_room

    report = fp.up_report(make_room(), make_poses())
    assert report["axis"] == 1
    assert report["source"] == "poses+floor"
    assert report["confidence"] > 0.8
    assert report["levelled"] is True


def test_up_report_flags_an_unlevelled_room():
    """The plan is axis-aligned, so a tilted room cannot be drawn correctly --
    and must say so rather than render a sheared picture silently."""
    from test_level import make_poses, make_room

    up = np.array([0.1, -0.94, 0.32])
    up /= np.linalg.norm(up)
    report = fp.up_report(make_room(up), make_poses(up))

    assert report["levelled"] is False
    assert report["snap_deg"] > 10
    assert report["confidence"] < 0.4
    assert any("level" in note for note in report["notes"])


def test_estimate_up_axis_reports_low_confidence_on_a_cube():
    """A cube has no flat direction; the UI must be told the guess is weak."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(5000, 3))
    assert fp.estimate_up_axis(pts)[2] < 0.3


def test_plan_transform_round_trips_within_a_pixel():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, 0, -3], [2, 2, 3], size=(2000, 3))
    t = fp.build_transform(pts, up_axis=1, size=600)

    px = t.world_to_pixel(pts)
    back = t.pixel_to_world(px)
    a, b = t.plane_axes
    assert np.allclose(back[:, 0], pts[:, a], atol=1.0 / t.scale)
    assert np.allclose(back[:, 1], pts[:, b], atol=1.0 / t.scale)


def test_plan_transform_puts_all_points_inside_the_image():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, 0, -3], [2, 2, 3], size=(2000, 3))
    t = fp.build_transform(pts, up_axis=1, size=600)
    px = t.world_to_pixel(pts)

    assert px[:, 0].min() >= -0.5 and px[:, 0].max() <= t.width + 0.5
    assert px[:, 1].min() >= -0.5 and px[:, 1].max() <= t.height + 0.5


def test_render_plan_shape_and_dtype():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, 0, -2], [2, 2, 2], size=(4000, 3))
    cols = rng.integers(0, 256, (4000, 3), dtype=np.uint8)
    t = fp.build_transform(pts, up_axis=1, size=200)

    img = fp.render_plan(pts, cols, t)
    assert img.shape == (t.height, t.width, 3)
    assert img.dtype == np.uint8


def test_object_footprints_produce_hulls():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, 0, -2], [2, 2, 2], size=(1000, 3))
    t = fp.build_transform(pts, up_axis=1, size=300)

    objs = [{"id": "obj_000", "label": "sofa", "centroid": [0, 0.4, 0],
             "obb": {"center": [0, 0.4, 0], "extent": [1, 0.5, 0.5],
                     "R": np.eye(3).tolist()}}]
    feet = fp.object_footprints(objs, t)
    assert len(feet) == 1 and len(feet[0]["hull"]) >= 3


def test_object_footprints_skip_malformed_objects():
    t = fp.build_transform(np.zeros((10, 3)) + [[1, 2, 3]], up_axis=1)
    assert fp.object_footprints([{"id": "x", "label": "y"}], t) == []


# --- endpoints --------------------------------------------------------------


def test_list_rooms(client):
    body = client.get("/api/rooms").json()
    assert [r["name"] for r in body["rooms"]] == ["demo"]
    assert body["rooms"][0]["artifacts"]["scene.ply"] is True


def test_room_detail_includes_camera_span(client):
    body = client.get("/api/rooms/demo").json()
    assert body["trajectory"]["span_max"] == pytest.approx(1.5)
    assert body["n_frames"] == 4


def test_unknown_room_404s(client):
    assert client.get("/api/rooms/nope").status_code == 404


def test_path_traversal_is_rejected(client):
    assert client.get("/api/rooms/..%2F..%2Fetc/objects").status_code in (400, 404)


def test_cloud_endpoint_returns_binary_with_count_header(client):
    r = client.get("/api/rooms/demo/cloud?max_points=1000")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert int(r.headers["X-Point-Count"]) == 1000
    assert len(r.content) == 4 + 1000 * 12 + 1000 * 3


def test_frames_endpoints(client):
    assert len(client.get("/api/rooms/demo/frames").json()["frames"]) == 4
    assert client.get("/api/rooms/demo/frames/0").headers["content-type"] == "image/png"
    assert client.get("/api/rooms/demo/frames/99").status_code == 404


def test_trajectory_endpoint(client):
    body = client.get("/api/rooms/demo/trajectory").json()
    assert len(body["poses"]) == 4
    assert body["stats"]["span_max"] == pytest.approx(1.5)


def test_pose_matrices_round_trip_the_trajectory_file(tmp_path):
    """The up estimator reads orientations from here, so a transposed or
    mis-normalised quaternion would quietly tilt every floor plan."""
    from room3d.level import rotation_between

    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, :3, :3] = rotation_between([0.0, 1.0, 0.0], [0.2, 0.9, -0.3])
    poses[:, :3, 3] = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]]
    save_trajectory_tum(tmp_path / "t.txt", poses)

    back = R.pose_matrices(tmp_path / "t.txt")
    assert back.shape == (3, 4, 4)
    assert np.allclose(back, poses, atol=1e-6)


def test_pose_matrices_tolerates_a_missing_trajectory(tmp_path):
    assert R.pose_matrices(tmp_path / "nope.txt") is None


def test_floorplan_meta_and_png(client):
    meta = client.get("/api/rooms/demo/floorplan/meta").json()
    assert meta["transform"]["width"] > 0
    assert "confidence" in meta["up_estimate"]
    # Poses were used, not just the cloud's shape.
    assert meta["up_estimate"]["source"].startswith("poses")
    assert meta["level"] is None                 # this fixture was never levelled

    png = client.get("/api/rooms/demo/floorplan.png?size=200")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_floorplan_meta_honours_a_manual_up_override(client):
    meta = client.get("/api/rooms/demo/floorplan/meta?up=x").json()
    assert meta["up_estimate"]["axis"] == "x"
    assert meta["up_estimate"]["source"] == "manual"
    assert meta["transform"]["plane_axes"] == ["y", "z"]


def test_refuse_endpoint_changes_counts(client):
    tight = client.post("/api/rooms/demo/refuse",
                        json={"radius_floor": 0.05}).json()
    loose = client.post("/api/rooms/demo/refuse",
                        json={"radius_floor": 5.0}).json()
    assert len(loose["objects"]) <= len(tight["objects"])
    assert tight["n_observations"] == 4


def test_refuse_does_not_save_unless_asked(client, room_dir):
    before = (room_dir / "objects.json").read_text()
    client.post("/api/rooms/demo/refuse", json={"radius_floor": 1.0})
    assert (room_dir / "objects.json").read_text() == before


def test_refuse_saves_when_asked(client, room_dir):
    r = client.post("/api/rooms/demo/refuse", json={"radius_floor": 1.0, "save": True})
    assert r.json()["saved"] is True
    assert (room_dir / "objects.prev.json").exists()
    assert json.loads((room_dir / "objects.json").read_text())["objects"]


def test_refuse_rejects_nonsense_parameters(client):
    assert client.post("/api/rooms/demo/refuse",
                       json={"radius_floor": -1}).status_code == 422


def test_partial_room_does_not_500(client, room_dir):
    """A run that died before labeling is exactly what you want to inspect."""
    (room_dir / "objects.json").unlink()
    (room_dir / "observations.json").unlink()

    assert client.get("/api/rooms").status_code == 200
    assert client.get("/api/rooms/demo").status_code == 200
    assert client.get("/api/rooms/demo/objects").status_code == 404
    assert client.get("/api/rooms/demo/refuse").status_code in (404, 405)


def test_index_and_static_are_served(client):
    assert client.get("/").status_code == 200
    assert "room3d" in client.get("/").text


def test_run_rejects_source_outside_the_project(client):
    r = client.post("/api/runs", json={"source": "../../etc/passwd", "room": "x"})
    assert r.status_code in (400, 404)


def test_run_rejects_bad_room_name(client, room_dir):
    r = client.post("/api/runs", json={"source": "out/demo/scene.ply", "room": "../evil"})
    assert r.status_code == 400
