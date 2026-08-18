"""Lift 2D detections into 3D.

This is cheap only because of a property of the reconstruction: global alignment
returns, for every input image, a per-pixel 3D point already expressed in the
world frame. So a detection's pixel set indexes straight into `pts3d[i]` and no
raycasting is involved.

The three things that actually go wrong here, and the defence against each:

1. Depth bleed. A mask's edge straddles the discontinuity between the object and
   the wall behind it, so a mean centroid is dragged backwards. Defence: erode
   the mask, then keep only the dominant front depth cluster, then take a median.
2. Bad geometry. Confidence is poor on textureless walls and glass. Defence:
   intersect with the reconstruction's own confidence mask and drop the detection
   outright if too little survives, rather than emit a confident wrong position.
3. Resolution mismatch. The VLM sees one image size, `pts3d` is another. Defence:
   one descaling helper, unit-tested, used by everything.

Everything here is a pure function over arrays. No LLM, no I/O.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import cv2
import numpy as np

# Gemini emits box coordinates normalised to this range.
GEMINI_COORD_SCALE = 1000.0


@dataclass
class OrientedBox:
    center: np.ndarray   # (3,)
    extent: np.ndarray   # (3,) full side lengths along the box axes
    R: np.ndarray        # (3, 3) columns are the box axes, right-handed

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.extent))

    def as_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "extent": self.extent.tolist(),
            "R": self.R.tolist(),
        }


@dataclass
class Observation:
    """One object, seen in one frame, lifted to 3D.

    `box_px` is carried through from the detection so the viewer can draw the
    object back onto the frame it came from. It is provenance, not input to any
    computation here.

    `points` are the surviving 3D points, subsampled. They exist so fusion can
    fit the final box to the *union* of what every frame saw. Without them a
    fused box can only ever be a copy of one frame's box or an average of
    quantities expressed in different axis frames, and the second of those is
    meaningless. They are deliberately not serialised into `as_dict`: one
    observation's points outweigh the whole JSON document.
    """
    frame_idx: int
    label: str
    vlm_confidence: float
    centroid: np.ndarray      # (3,) robust median
    obb: OrientedBox
    n_points: int
    support: float            # fraction of the detection's pixels that survived
    box_px: tuple[int, int, int, int] | None = None    # x0, y0, x1, y1
    points: np.ndarray | None = None                   # (M, 3), M <= max_points

    def as_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "label": self.label,
            "vlm_confidence": round(float(self.vlm_confidence), 4),
            "centroid": [round(float(v), 4) for v in self.centroid],
            "obb": self.obb.as_dict(),
            "n_points": int(self.n_points),
            "support": round(float(self.support), 4),
            "box_px": list(self.box_px) if self.box_px else None,
        }


def descale_box(box_2d, height: int, width: int) -> tuple[int, int, int, int]:
    """Gemini `[ymin, xmin, ymax, xmax]` in 0-1000 -> pixel `(x0, y0, x1, y1)`.

    Note the axis order swap: Gemini puts y first, pixel convention puts x first.
    Getting this wrong produces plausible-looking but transposed results, which
    is why it lives in one tested function instead of being inlined.
    """
    if len(box_2d) != 4:
        raise ValueError(f"box_2d must have 4 elements, got {len(box_2d)}")

    ymin, xmin, ymax, xmax = (float(v) for v in box_2d)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    x0 = int(np.floor(xmin / GEMINI_COORD_SCALE * width))
    x1 = int(np.ceil(xmax / GEMINI_COORD_SCALE * width))
    y0 = int(np.floor(ymin / GEMINI_COORD_SCALE * height))
    y1 = int(np.ceil(ymax / GEMINI_COORD_SCALE * height))

    x0 = int(np.clip(x0, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    x1 = int(np.clip(x1, x0 + 1, width))
    y1 = int(np.clip(y1, y0 + 1, height))
    return x0, y0, x1, y1


def box_to_mask(box_px: tuple[int, int, int, int], height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = box_px
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def decode_gemini_mask(
    mask_b64: str,
    box_px: tuple[int, int, int, int],
    height: int,
    width: int,
    threshold: int = 127,
) -> np.ndarray:
    """Gemini's per-object mask is a base64 PNG probability map covering the box.

    Returns a full-image boolean mask.
    """
    from PIL import Image

    if mask_b64.startswith("data:"):
        mask_b64 = mask_b64.split(",", 1)[1]

    raw = base64.b64decode(mask_b64)
    probs = np.array(Image.open(io.BytesIO(raw)).convert("L"))

    x0, y0, x1, y1 = box_px
    bh, bw = y1 - y0, x1 - x0
    if probs.shape != (bh, bw):
        probs = cv2.resize(probs, (bw, bh), interpolation=cv2.INTER_LINEAR)

    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = probs >= threshold
    return mask


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Pull the mask in from its edges, where depth bleed lives.

    If erosion would erase the mask entirely (thin objects like a monitor bezel),
    the original is kept — a slightly bled centroid beats no detection.
    """
    if px <= 0:
        return mask
    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return eroded if eroded.any() else mask


def dominant_depth_cluster(depths: np.ndarray, eps: float) -> np.ndarray:
    """Index mask selecting the dominant cluster of a 1-D depth distribution.

    Equivalent to DBSCAN with min_samples=1 on sorted values: split wherever the
    gap between consecutive depths exceeds `eps`, then keep the largest run. Ties
    go to the nearer cluster, because the object is in front of its background.
    """
    if depths.size == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(depths)
    ordered = depths[order]
    split_at = np.nonzero(np.diff(ordered) > eps)[0] + 1
    runs = np.split(np.arange(ordered.size), split_at)

    best = max(runs, key=lambda r: (r.size, -ordered[r[0]]))

    keep = np.zeros(depths.size, dtype=bool)
    keep[order[best]] = True
    return keep


def fit_oriented_box(points: np.ndarray) -> OrientedBox:
    """PCA-aligned bounding box around `points` (M, 3)."""
    if points.shape[0] < 3:
        center = points.mean(axis=0) if points.size else np.zeros(3)
        return OrientedBox(center, np.zeros(3), np.eye(3))

    mean = points.mean(axis=0)
    centred = points - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    R = vt.T

    if np.linalg.det(R) < 0:      # keep it a rotation, not a reflection
        R[:, 2] *= -1

    local = centred @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    extent = hi - lo
    center = mean + R @ ((lo + hi) / 2.0)
    return OrientedBox(center, extent, R)


def box_corners(obb: OrientedBox) -> np.ndarray:
    """The eight corners of an oriented box, in world coordinates. (8, 3)."""
    signs = np.array(
        [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]
    )
    return (signs * obb.extent[None, :]) @ obb.R.T + obb.center[None, :]


def points_inside_fraction(obb: OrientedBox, points: np.ndarray, tol: float = 1e-6) -> float:
    """Share of `points` lying inside the box. The coherence check on any fit.

    A box whose centre, extent and rotation came from different sources can look
    plausible in isolation and still contain almost none of the geometry it
    claims to describe, which is the only thing that actually matters.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return 1.0
    local = (pts - obb.center[None, :]) @ obb.R
    half = obb.extent / 2.0 + tol
    return float((np.abs(local) <= half[None, :]).all(axis=1).mean())


def _ground_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An orthonormal frame with `up` as its second axis."""
    u = np.asarray(up, dtype=np.float64).reshape(3)
    u = u / max(np.linalg.norm(u), 1e-12)

    seed = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = seed - (seed @ u) * u
    e1 /= max(np.linalg.norm(e1), 1e-12)
    return u, e1, np.cross(u, e1)


def fit_gravity_aligned_box(
    points: np.ndarray,
    up: np.ndarray,
    *,
    percentile: float = 1.0,
    yaw_step_deg: float = 1.0,
) -> OrientedBox:
    """Minimum-footprint box around `points` with one axis locked to `up`.

    Three deliberate departures from a plain PCA box:

    - **Gravity, not the data, fixes the vertical axis.** PCA on a partial view
      finds the axes of the *visible surface* -- a sofa seen from the front is a
      slab, and its principal axes describe the slab, not the sofa. Furniture
      stands on a floor, so only the yaw is genuinely unknown.
    - **Yaw comes from the smallest footprint**, swept at `yaw_step_deg`. This is
      rotating calipers with a fixed step, which is the right trade here: the
      exact-minimum algorithm needs a convex hull and buys nothing at the
      accuracy the point clouds support.
    - **Extents come from percentiles, not min/max**, so the handful of pixels
      that leaked onto the wall behind the object cannot stretch the box across
      the room. `percentile=0` reproduces min/max exactly.

    The returned rotation has `up` as column 1 and is right-handed.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]

    u, e1, e2 = _ground_basis(up)
    if len(pts) == 0:
        return OrientedBox(np.zeros(3), np.zeros(3), np.column_stack([e1, u, np.cross(e1, u)]))

    lo_q, hi_q = float(percentile), 100.0 - float(percentile)

    a, b = pts @ e1, pts @ e2
    angles = np.radians(np.arange(0.0, 90.0, max(yaw_step_deg, 1e-3)))
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]

    # (n_angles, n_points): every candidate yaw evaluated at once. The two rows
    # are the coordinates along `right` and `forward` below -- the signs have to
    # match those axes exactly, or the extents come out right and the centre
    # lands somewhere else entirely.
    x = a[None, :] * cos + b[None, :] * sin
    z = a[None, :] * sin - b[None, :] * cos

    x_lo, x_hi = np.percentile(x, [lo_q, hi_q], axis=1)
    z_lo, z_hi = np.percentile(z, [lo_q, hi_q], axis=1)
    best = int(np.argmin((x_hi - x_lo) * (z_hi - z_lo)))

    theta = angles[best]
    right = np.cos(theta) * e1 + np.sin(theta) * e2
    forward = np.cross(right, u)              # right x up = forward keeps det(R) = +1

    y = pts @ u
    y_lo, y_hi = np.percentile(y, [lo_q, hi_q])

    extent = np.array([x_hi[best] - x_lo[best], y_hi - y_lo, z_hi[best] - z_lo[best]])
    center = (
        right * (x_lo[best] + x_hi[best]) / 2.0
        + u * (y_lo + y_hi) / 2.0
        + forward * (z_lo[best] + z_hi[best]) / 2.0
    )
    return OrientedBox(center, np.maximum(extent, 0.0), np.column_stack([right, u, forward]))


def snap_to_floor(
    obb: OrientedBox,
    floor_height: float,
    up: np.ndarray,
    *,
    threshold: float = 0.15,
) -> OrientedBox:
    """Pull a box's underside onto the floor when it is already nearly there.

    Only the *visible* surface of a sofa is ever reconstructed, so its box floats
    a few centimetres above the ground or sinks a few below it, depending on
    which way the depth bled. Snapping fixes both without inventing anything: a
    box already within `threshold` of the floor is meant to be resting on it. A
    shelf on a wall is nowhere near, so it is left exactly as it was.
    """
    u = np.asarray(up, dtype=np.float64).reshape(3)
    u = u / max(np.linalg.norm(u), 1e-12)

    alignment = obb.R.T @ u
    axis = int(np.argmax(np.abs(alignment)))
    if abs(alignment[axis]) < 0.99:
        # No box axis is vertical, so "the underside" is not a face of this box
        # and there is nothing well-defined to snap.
        return obb

    half = float(obb.extent[axis]) / 2.0
    height = float(obb.center @ u)
    bottom, top = height - half, height + half

    if abs(bottom - floor_height) > threshold:
        return obb

    extent = obb.extent.copy()
    extent[axis] = max(top - floor_height, 0.0)
    center = obb.center + ((top + floor_height) / 2.0 - height) * u
    return OrientedBox(center, extent, obb.R.copy())


def project_detection(
    mask: np.ndarray,
    pts3d: np.ndarray,
    conf_mask: np.ndarray,
    camera_center: np.ndarray,
    *,
    frame_idx: int,
    label: str,
    vlm_confidence: float,
    erode_px: int = 3,
    depth_eps: float = 0.15,
    min_points: int = 40,
    box_px: tuple[int, int, int, int] | None = None,
    up: np.ndarray | None = None,
    max_points: int = 2048,
    obb_percentile: float = 1.0,
) -> Observation | None:
    """Lift one 2D detection to a 3D observation, or None if unsupported.

    Returning None is a feature: a detection over a low-confidence wall should
    disappear rather than contribute a fabricated 3D position.

    `up` switches the box fit from PCA to gravity-aligned. Pass it whenever the
    reconstruction has been levelled -- which it is by default -- because a PCA
    box on a partial view describes the visible surface rather than the object.
    """
    if mask.shape != conf_mask.shape:
        raise ValueError(f"mask {mask.shape} does not match conf_mask {conf_mask.shape}")
    if pts3d.shape[:2] != mask.shape:
        raise ValueError(f"pts3d {pts3d.shape[:2]} does not match mask {mask.shape}")

    n_requested = int(mask.sum())
    if n_requested == 0:
        return None

    selected = erode_mask(mask, erode_px) & conf_mask
    if selected.sum() < min_points:
        return None

    points = pts3d[selected]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] < min_points:
        return None

    depths = np.linalg.norm(points - camera_center[None, :], axis=1)
    keep = dominant_depth_cluster(depths, depth_eps)
    points = points[keep]
    if points.shape[0] < min_points:
        return None

    n_kept = int(points.shape[0])
    obb = (
        fit_oriented_box(points)
        if up is None
        else fit_gravity_aligned_box(points, up, percentile=obb_percentile)
    )

    return Observation(
        frame_idx=frame_idx,
        label=label,
        vlm_confidence=vlm_confidence,
        centroid=np.median(points, axis=0),
        obb=obb,
        n_points=n_kept,
        support=float(n_kept) / float(n_requested),
        box_px=box_px,
        points=subsample(points, max_points),
    )


def subsample(points: np.ndarray, limit: int, seed: int = 0) -> np.ndarray:
    """At most `limit` points, chosen without replacement and deterministically.

    Fusion pools points across every frame that saw an object, so an uncapped
    detection covering half the image would dominate both memory and the fit.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if limit <= 0 or len(pts) <= limit:
        return pts
    idx = np.random.default_rng(seed).choice(len(pts), limit, replace=False)
    return pts[np.sort(idx)]


def camera_center_from_pose(pose_c2w: np.ndarray) -> np.ndarray:
    """Camera-to-world 4x4 -> camera origin in world coordinates."""
    return np.asarray(pose_c2w, dtype=np.float64)[:3, 3]
