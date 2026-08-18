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


def test_a_point_at_the_camera_center_is_not_visible(recwarn):
    """Divides by zero in project_to_frame, coming back nan/not-in-view.

    Regression: the clipped pixel lookup used to cast that nan straight to an
    integer index and crash with an IndexError instead of returning False.
    """
    recon = make_recon(flat_wall(2.0))
    uv, seen = visible_in_frame(np.array([[0.0, 0.0, 0.0]]), recon, 0)
    assert not seen[0]
    assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)
