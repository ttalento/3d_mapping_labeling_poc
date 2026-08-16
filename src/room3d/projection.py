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
    """
    frame_idx: int
    label: str
    vlm_confidence: float
    centroid: np.ndarray      # (3,) robust median
    obb: OrientedBox
    n_points: int
    support: float            # fraction of the detection's pixels that survived
    box_px: tuple[int, int, int, int] | None = None    # x0, y0, x1, y1

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
) -> Observation | None:
    """Lift one 2D detection to a 3D observation, or None if unsupported.

    Returning None is a feature: a detection over a low-confidence wall should
    disappear rather than contribute a fabricated 3D position.
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

    return Observation(
        frame_idx=frame_idx,
        label=label,
        vlm_confidence=vlm_confidence,
        centroid=np.median(points, axis=0),
        obb=fit_oriented_box(points),
        n_points=int(points.shape[0]),
        support=float(points.shape[0]) / float(n_requested),
        box_px=box_px,
    )


def camera_center_from_pose(pose_c2w: np.ndarray) -> np.ndarray:
    """Camera-to-world 4x4 -> camera origin in world coordinates."""
    return np.asarray(pose_c2w, dtype=np.float64)[:3, 3]
