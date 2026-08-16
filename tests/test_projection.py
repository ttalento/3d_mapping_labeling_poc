"""Gate 4: projection correctness against a synthetic scene with known geometry.

The scene is built the way the reconstruction stage delivers it -- a per-pixel
world-frame pointmap plus a confidence mask -- so these tests exercise the same
code path as a real run, minus the network.
"""

import numpy as np
import pytest

from room3d.projection import (
    camera_center_from_pose,
    descale_box,
    dominant_depth_cluster,
    erode_mask,
    fit_oriented_box,
    project_detection,
)

H, W = 64, 96
CAMERA_CENTER = np.zeros(3)

# A 20x20 pixel object 2 m in front of the camera, on a wall 5 m behind it.
OBJECT_SLICE = (slice(20, 40), slice(30, 50))
OBJECT_DEPTH = 2.0
WALL_DEPTH = 5.0


def make_scene():
    """Pointmap where the object patch sits at z=2 and everything else at z=5."""
    pts3d = np.zeros((H, W, 3), dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)

    pts3d[..., 0] = gx
    pts3d[..., 1] = gy
    pts3d[..., 2] = WALL_DEPTH
    pts3d[OBJECT_SLICE][..., 2] = OBJECT_DEPTH

    conf = np.ones((H, W), dtype=bool)
    return pts3d, conf


def object_mask():
    mask = np.zeros((H, W), dtype=bool)
    mask[OBJECT_SLICE] = True
    return mask


# --- descale_box: the transposition trap -------------------------------------


def test_descale_box_maps_full_range_to_full_image():
    assert descale_box([0, 0, 1000, 1000], H, W) == (0, 0, W, H)


def test_descale_box_respects_y_first_ordering():
    # ymin=0, xmin=500, ymax=500, xmax=1000  ->  right half, top half
    x0, y0, x1, y1 = descale_box([0, 500, 500, 1000], H, W)
    assert (x0, x1) == (W // 2, W)
    assert (y0, y1) == (0, H // 2)


def test_descale_box_is_not_transposed():
    """A wide-flat box must stay wide and flat, never tall and thin."""
    x0, y0, x1, y1 = descale_box([400, 0, 600, 1000], H, W)
    assert (x1 - x0) > (y1 - y0)


def test_descale_box_swaps_inverted_input():
    assert descale_box([1000, 1000, 0, 0], H, W) == (0, 0, W, H)


def test_descale_box_clamps_out_of_range():
    x0, y0, x1, y1 = descale_box([-50, -50, 5000, 5000], H, W)
    assert (x0, y0) == (0, 0) and (x1, y1) == (W, H)


def test_descale_box_rejects_wrong_length():
    with pytest.raises(ValueError):
        descale_box([0, 0, 100], H, W)


# --- depth clustering --------------------------------------------------------


def test_dominant_depth_cluster_picks_larger_group():
    depths = np.array([2.0, 2.01, 2.02, 2.03, 5.0, 5.01])
    keep = dominant_depth_cluster(depths, eps=0.15)
    assert keep.tolist() == [True, True, True, True, False, False]


def test_dominant_depth_cluster_prefers_nearer_on_tie():
    depths = np.array([2.0, 2.01, 5.0, 5.01])
    keep = dominant_depth_cluster(depths, eps=0.15)
    assert keep.tolist() == [True, True, False, False]


def test_dominant_depth_cluster_handles_empty():
    assert dominant_depth_cluster(np.array([]), eps=0.1).size == 0


# --- the headline test -------------------------------------------------------


def test_projected_centroid_matches_ground_truth():
    """Gate 4: centroid within 1 cm of truth."""
    pts3d, conf = make_scene()
    obs = project_detection(
        object_mask(), pts3d, conf, CAMERA_CENTER,
        frame_idx=0, label="monitor", vlm_confidence=0.9,
        erode_px=2, depth_eps=0.15, min_points=10,
    )
    assert obs is not None
    truth = pts3d[OBJECT_SLICE].reshape(-1, 3).mean(axis=0)
    assert np.linalg.norm(obs.centroid - truth) < 0.01


def test_depth_bleed_does_not_drag_centroid_to_the_wall():
    """A mask overshooting onto the background must still land on the object.

    This is the failure this module exists to prevent: without depth clustering
    the centroid drifts toward 5 m.
    """
    pts3d, conf = make_scene()
    bled = object_mask()
    bled[15:45, 25:55] = True          # 5 px of overshoot on every side

    obs = project_detection(
        bled, pts3d, conf, CAMERA_CENTER,
        frame_idx=0, label="monitor", vlm_confidence=0.9,
        erode_px=2, depth_eps=0.15, min_points=10,
    )
    assert obs is not None
    assert abs(obs.centroid[2] - OBJECT_DEPTH) < 0.05, "centroid bled toward the wall"


def test_low_confidence_geometry_drops_the_detection():
    """Better to emit nothing than a fabricated 3D position."""
    pts3d, conf = make_scene()
    conf[:] = False
    assert project_detection(
        object_mask(), pts3d, conf, CAMERA_CENTER,
        frame_idx=0, label="monitor", vlm_confidence=0.9, min_points=10,
    ) is None


def test_too_few_surviving_points_drops_the_detection():
    pts3d, conf = make_scene()
    tiny = np.zeros((H, W), dtype=bool)
    tiny[30:32, 40:42] = True
    assert project_detection(
        tiny, pts3d, conf, CAMERA_CENTER,
        frame_idx=0, label="mug", vlm_confidence=0.9, erode_px=0, min_points=40,
    ) is None


def test_empty_mask_returns_none():
    pts3d, conf = make_scene()
    assert project_detection(
        np.zeros((H, W), bool), pts3d, conf, CAMERA_CENTER,
        frame_idx=0, label="x", vlm_confidence=0.5,
    ) is None


def test_nonfinite_points_are_ignored():
    pts3d, conf = make_scene()
    pts3d[25, 35] = np.nan
    pts3d[26, 36] = np.inf
    obs = project_detection(
        object_mask(), pts3d, conf, CAMERA_CENTER,
        frame_idx=0, label="monitor", vlm_confidence=0.9, erode_px=1, min_points=10,
    )
    assert obs is not None and np.isfinite(obs.centroid).all()


def test_shape_mismatch_is_an_error():
    pts3d, conf = make_scene()
    with pytest.raises(ValueError):
        project_detection(
            np.zeros((H + 1, W), bool), pts3d, conf, CAMERA_CENTER,
            frame_idx=0, label="x", vlm_confidence=0.5,
        )


# --- helpers -----------------------------------------------------------------


def test_erode_mask_keeps_thin_objects_alive():
    thin = np.zeros((H, W), dtype=bool)
    thin[30, 20:60] = True                      # 1 px tall
    assert erode_mask(thin, 3).any(), "erosion erased a thin object"


def test_fit_oriented_box_recovers_extent_exactly_on_a_regular_grid():
    """A symmetric grid gives an exactly diagonal covariance, so the PCA axes are
    exactly the box axes and the extent must come back exact."""
    gx, gy, gz = np.meshgrid(
        np.linspace(-1, 1, 12), np.linspace(-2, 2, 12), np.linspace(-3, 3, 12),
        indexing="ij",
    )
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    box = fit_oriented_box(pts)
    assert np.allclose(box.center, 0, atol=1e-6)
    assert np.allclose(np.sort(box.extent), [2, 4, 6], atol=1e-6)
    assert np.linalg.det(box.R) > 0, "rotation matrix must be right-handed"


def test_fit_oriented_box_on_a_random_cloud_is_approximately_right():
    """PCA axes on a finite random sample sit a few degrees off the true axes,
    which inflates the extent slightly. Tolerance reflects that; the box is an
    approximate bounding region, not a metrology instrument."""
    pts = np.random.default_rng(0).uniform([-1, -2, -3], [1, 2, 3], size=(2000, 3))
    box = fit_oriented_box(pts)
    assert np.allclose(box.center, 0, atol=0.15)
    assert np.allclose(np.sort(box.extent), [2, 4, 6], rtol=0.12)
    assert np.linalg.det(box.R) > 0


def test_fit_oriented_box_handles_degenerate_input():
    box = fit_oriented_box(np.zeros((2, 3)))
    assert np.allclose(box.extent, 0)
    assert np.allclose(box.R, np.eye(3))


def test_camera_center_from_pose():
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(camera_center_from_pose(pose), [1.0, 2.0, 3.0])
