"""Gate 4b: the 3D box must be fit to points, level with gravity, and coherent.

The bug this file exists to prevent: a box whose centre, extent and rotation came
from three different sources, so one corner landed on the object and the rest did
not. Every test here asserts the box against the points it claims to describe.
"""

import numpy as np
import pytest

from room3d.level import UP_VECTOR
from room3d.projection import (
    OrientedBox,
    box_corners,
    fit_gravity_aligned_box,
    points_inside_fraction,
    project_detection,
    snap_to_floor,
)


def cuboid_points(centre, size, yaw_deg=0.0, n=4000, seed=0, up=UP_VECTOR):
    """Points filling a box of `size`, yawed by `yaw_deg` about `up`."""
    rng = np.random.default_rng(seed)
    local = (rng.random((n, 3)) - 0.5) * np.asarray(size, float)
    t = np.radians(yaw_deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return local @ R.T + np.asarray(centre, float)


# --- the box is level --------------------------------------------------------


def test_one_box_axis_is_exactly_the_up_vector():
    obb = fit_gravity_aligned_box(cuboid_points([0, 1, 0], (2.0, 0.8, 0.9)), UP_VECTOR)
    assert np.allclose(obb.R[:, 1], UP_VECTOR, atol=1e-9)


def test_rotation_stays_a_right_handed_orthonormal_frame():
    obb = fit_gravity_aligned_box(
        cuboid_points([1, 0.5, -2], (2.0, 0.8, 0.9), yaw_deg=37.0), UP_VECTOR
    )
    assert np.allclose(obb.R.T @ obb.R, np.eye(3), atol=1e-9)
    assert np.linalg.det(obb.R) == pytest.approx(1.0)


def test_works_for_an_arbitrary_up_vector_not_just_plus_y():
    up = np.array([0.1, 0.9, 0.2])
    up = up / np.linalg.norm(up)
    obb = fit_gravity_aligned_box(cuboid_points([0, 0, 0], (2.0, 0.8, 0.9)), up)
    assert np.allclose(obb.R[:, 1], up, atol=1e-9)


# --- the box actually fits the points ----------------------------------------


def test_recovers_the_extents_of_a_yawed_cuboid():
    """A sofa-shaped box turned 30 degrees must come back 2.0 x 0.8 x 0.9,
    not the larger axis-aligned envelope that a naive fit would report."""
    obb = fit_gravity_aligned_box(
        cuboid_points([0, 0.45, 0], (2.0, 0.8, 0.9), yaw_deg=30.0, n=20000),
        UP_VECTOR,
        percentile=0.0,
    )
    assert sorted(np.round(obb.extent, 1)) == pytest.approx([0.8, 0.9, 2.0], abs=0.06)


def test_recovers_the_yaw_of_a_rotated_cuboid():
    obb = fit_gravity_aligned_box(
        cuboid_points([0, 0, 0], (2.0, 0.8, 0.9), yaw_deg=30.0, n=20000), UP_VECTOR
    )
    long_axis = obb.R[:, int(np.argmax(obb.extent))]
    truth = np.array([np.cos(np.radians(30.0)), 0.0, -np.sin(np.radians(30.0))])
    assert abs(float(long_axis @ truth)) == pytest.approx(1.0, abs=0.02)


def test_the_points_it_was_fit_to_lie_inside_it():
    """The coherence check the old fusion code failed."""
    pts = cuboid_points([1.5, 0.4, -2.0], (2.1, 0.85, 0.95), yaw_deg=17.0)
    obb = fit_gravity_aligned_box(pts, UP_VECTOR, percentile=0.0)
    assert points_inside_fraction(obb, pts) == pytest.approx(1.0, abs=1e-6)


def test_the_centre_is_the_centre_of_the_extent_not_the_mean_of_the_points():
    """Half the points crowd one end; the box must still be centred on the box."""
    dense = cuboid_points([0, 0, 0], (0.2, 0.8, 0.9), n=8000, seed=1)
    sparse = cuboid_points([1.5, 0, 0], (0.2, 0.8, 0.9), n=200, seed=2)
    pts = np.vstack([dense, sparse])

    obb = fit_gravity_aligned_box(pts, UP_VECTOR, percentile=0.0)
    assert obb.center[0] == pytest.approx(0.75, abs=0.06)
    assert float(pts[:, 0].mean()) < 0.2         # the mean would have been wrong


# --- percentile trimming rejects depth bleed ---------------------------------


def test_percentile_trimming_ignores_a_few_bled_points():
    """A handful of pixels leaking onto the wall 4 m behind must not stretch the box."""
    pts = cuboid_points([0, 0.45, 0], (2.0, 0.8, 0.9), n=5000)
    bled = pts[:80].copy()
    bled[:, 2] += 4.0
    obb = fit_gravity_aligned_box(np.vstack([pts, bled]), UP_VECTOR, percentile=2.0)
    assert obb.diagonal < 2.6


def test_percentile_zero_keeps_every_point():
    pts = cuboid_points([0, 0, 0], (1.0, 1.0, 1.0), n=2000)
    trimmed = fit_gravity_aligned_box(pts, UP_VECTOR, percentile=10.0)
    full = fit_gravity_aligned_box(pts, UP_VECTOR, percentile=0.0)
    assert full.diagonal > trimmed.diagonal


def test_degenerate_input_does_not_raise():
    for pts in (np.zeros((0, 3)), np.zeros((1, 3)), np.ones((2, 3))):
        obb = fit_gravity_aligned_box(pts, UP_VECTOR)
        assert obb.extent.shape == (3,)
        assert np.isfinite(obb.center).all()


# --- floor snapping ----------------------------------------------------------


def test_a_box_hovering_just_above_the_floor_is_pulled_down_to_it():
    obb = OrientedBox(np.array([0.0, 0.55, 0.0]), np.array([2.0, 0.9, 0.9]), np.eye(3))
    snapped = snap_to_floor(obb, floor_height=0.0, up=UP_VECTOR, threshold=0.15)
    assert snapped.extent[1] == pytest.approx(1.0)
    assert snapped.center[1] == pytest.approx(0.5)


def test_a_box_sunk_below_the_floor_is_lifted_onto_it():
    obb = OrientedBox(np.array([0.0, 0.4, 0.0]), np.array([2.0, 0.9, 0.9]), np.eye(3))
    snapped = snap_to_floor(obb, floor_height=0.0, up=UP_VECTOR, threshold=0.15)
    assert snapped.extent[1] == pytest.approx(0.85)
    assert snapped.center[1] == pytest.approx(0.425)


def test_a_shelf_high_on_the_wall_is_left_alone():
    obb = OrientedBox(np.array([0.0, 1.8, 0.0]), np.array([1.2, 0.3, 0.3]), np.eye(3))
    snapped = snap_to_floor(obb, floor_height=0.0, up=UP_VECTOR, threshold=0.15)
    assert snapped.center[1] == pytest.approx(1.8)
    assert snapped.extent[1] == pytest.approx(0.3)


def test_snapping_finds_the_up_axis_wherever_it_sits_in_the_rotation():
    """The up axis is column 1 by construction, but nothing may depend on that."""
    R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    obb = OrientedBox(np.array([0.0, 0.55, 0.0]), np.array([0.9, 0.9, 2.0]), R)
    snapped = snap_to_floor(obb, floor_height=0.0, up=UP_VECTOR, threshold=0.15)
    assert snapped.extent[0] == pytest.approx(1.0)
    assert snapped.center[1] == pytest.approx(0.5)


# --- box_corners is the shared definition of "the box" -----------------------


def test_box_corners_returns_eight_corners_spanning_the_extent():
    obb = OrientedBox(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), np.eye(3))
    corners = box_corners(obb)
    assert corners.shape == (8, 3)
    assert np.allclose(corners.min(axis=0), [0.0, 0.0, 0.0])
    assert np.allclose(corners.max(axis=0), [2.0, 4.0, 6.0])


# --- observations keep their points so fusion can refit ----------------------


H, W = 64, 96


def _scene_with_object():
    pts3d = np.zeros((H, W, 3), dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    pts3d[..., 0], pts3d[..., 1], pts3d[..., 2] = gx, gy, 5.0
    pts3d[20:40, 30:50, 2] = 2.0

    mask = np.zeros((H, W), dtype=bool)
    mask[20:40, 30:50] = True
    return pts3d, np.ones((H, W), dtype=bool), mask


def test_project_detection_keeps_the_points_it_measured():
    pts3d, conf, mask = _scene_with_object()
    obs = project_detection(
        mask, pts3d, conf, np.zeros(3), frame_idx=0, label="box", vlm_confidence=0.9
    )
    assert obs.points is not None
    assert obs.points.shape[1] == 3
    assert len(obs.points) == obs.n_points


def test_retained_points_are_capped_so_fusion_stays_cheap():
    pts3d, conf, mask = _scene_with_object()
    obs = project_detection(
        mask, pts3d, conf, np.zeros(3), frame_idx=0, label="box",
        vlm_confidence=0.9, max_points=50,
    )
    assert len(obs.points) == 50
    assert obs.n_points > 50                 # the true count is still reported


def test_points_are_not_serialised_into_the_json_record():
    pts3d, conf, mask = _scene_with_object()
    obs = project_detection(
        mask, pts3d, conf, np.zeros(3), frame_idx=0, label="box", vlm_confidence=0.9
    )
    import json

    json.dumps(obs.as_dict())                # must not raise on a numpy array
    assert "points" not in obs.as_dict()


def test_projection_boxes_are_gravity_aligned_when_up_is_given():
    pts3d, conf, mask = _scene_with_object()
    obs = project_detection(
        mask, pts3d, conf, np.zeros(3), frame_idx=0, label="box",
        vlm_confidence=0.9, up=UP_VECTOR,
    )
    assert np.allclose(obs.obb.R[:, 1], UP_VECTOR, atol=1e-9)
