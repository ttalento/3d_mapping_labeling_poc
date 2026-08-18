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
