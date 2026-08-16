"""Top-down room map: the point cloud rasterised onto the floor plane.

Why this rather than a stitched panorama. Stitching assumes the camera rotates
about a single point; a walkthrough translates, so a panorama of one has no
consistent projection and the seams warp. A floor-plan projection is well
defined for a moving camera, and it is the view that actually answers "where is
the sofa relative to the window".

The hard part is that DUSt3R has no gravity vector, so nothing in the
reconstruction says which way is up. `room3d.level` recovers it from the camera
poses and a floor-plane fit; this module only has to consume the answer, report
how confident it is, and let the UI override -- silently guessing wrong would
produce a convincing picture of the wrong plane.

A `PlanTransform` is axis-aligned, so it can only render a room whose up
direction *is* a coordinate axis. That is why levelling is a pipeline step
rather than a viewer trick: on an unlevelled room the best this can do is snap
to the nearest axis and say how far off it had to reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..level import estimate_up

AXES = {"x": 0, "y": 1, "z": 2}

# Past this much snapping error, an axis-aligned plan is a sheared projection of
# the room rather than a picture of it, and the confidence should say so.
MAX_SNAP_DEG = 20.0


@dataclass
class PlanTransform:
    """Maps world XY (in the floor plane) to pixels, and back."""
    up_axis: int
    up_sign: float
    origin: tuple[float, float]      # world coords of pixel (0, 0)
    scale: float                     # pixels per metre
    width: int
    height: int

    @property
    def plane_axes(self) -> tuple[int, int]:
        return tuple(a for a in (0, 1, 2) if a != self.up_axis)

    def world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) world -> (N, 2) pixel."""
        a, b = self.plane_axes
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        u = (pts[:, a] - self.origin[0]) * self.scale
        v = self.height - 1 - (pts[:, b] - self.origin[1]) * self.scale
        return np.stack([u, v], axis=1)

    def pixel_to_world(self, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) pixel -> (N, 2) world coords on the two plane axes."""
        px = np.atleast_2d(np.asarray(pixels, dtype=float))
        x = px[:, 0] / self.scale + self.origin[0]
        y = (self.height - 1 - px[:, 1]) / self.scale + self.origin[1]
        return np.stack([x, y], axis=1)

    def as_dict(self) -> dict:
        return {
            "up_axis": "xyz"[self.up_axis],
            "up_sign": self.up_sign,
            "plane_axes": ["xyz"[a] for a in self.plane_axes],
            "origin": list(self.origin),
            "scale": self.scale,
            "width": self.width,
            "height": self.height,
        }


def up_report(points: np.ndarray, poses: np.ndarray | None = None) -> dict:
    """Which axis to plan against, and everything the UI needs to judge it.

    The underlying estimate is a free 3D vector; a plan needs an axis. Snapping
    to the nearest one is lossless on a levelled room (where up is exactly +Y)
    and lossy on a raw one, so the reported confidence is the estimator's own
    score scaled down by how far the snap had to move.
    """
    points = np.asarray(points)
    if len(points) < 10:
        return {
            "axis": 1, "sign": 1.0, "confidence": 0.0, "source": "none",
            "snap_deg": 0.0, "levelled": False,
            "notes": ["too few points to estimate"],
        }

    est = estimate_up(points, poses)
    axis = int(np.argmax(np.abs(est.up)))
    sign = float(np.sign(est.up[axis])) or 1.0

    snap = float(np.degrees(np.arccos(np.clip(abs(est.up[axis]), 0.0, 1.0))))
    penalty = float(np.clip(1.0 - snap / MAX_SNAP_DEG, 0.0, 1.0))
    levelled = snap < 1.0 and axis == 1 and sign > 0

    notes = list(est.notes)
    if not levelled and penalty < 0.9:
        notes.append(
            f"up is {snap:.0f} deg off the nearest axis, so this plan is sheared; "
            "run `room3d level --room <name>` to stand the room upright"
        )

    return {
        "axis": axis,
        "sign": sign,
        "confidence": round(est.confidence * penalty, 3),
        "source": est.source,
        "snap_deg": round(snap, 1),
        "levelled": levelled,
        "up": [round(float(v), 4) for v in est.up],
        "notes": notes,
    }


def estimate_up_axis(
    points: np.ndarray, poses: np.ndarray | None = None
) -> tuple[int, float, float]:
    """(axis, sign, confidence) -- the tuple form of `up_report`."""
    r = up_report(points, poses)
    return r["axis"], r["sign"], r["confidence"]


def build_transform(
    points: np.ndarray, up_axis: int, *, size: int = 900, margin: float = 0.04
) -> PlanTransform:
    a, b = (ax for ax in (0, 1, 2) if ax != up_axis)

    lo = np.array([points[:, a].min(), points[:, b].min()])
    hi = np.array([points[:, a].max(), points[:, b].max()])
    extent = np.maximum(hi - lo, 1e-6)

    pad = extent.max() * margin
    lo -= pad
    hi += pad
    extent = hi - lo

    scale = (size - 1) / extent.max()
    width = max(1, int(np.ceil(extent[0] * scale)) + 1)
    height = max(1, int(np.ceil(extent[1] * scale)) + 1)

    return PlanTransform(
        up_axis=up_axis, up_sign=1.0, origin=(float(lo[0]), float(lo[1])),
        scale=float(scale), width=width, height=height,
    )


def render_plan(
    points: np.ndarray,
    colors: np.ndarray,
    transform: PlanTransform,
    *,
    drop_ceiling: float = 0.15,
) -> np.ndarray:
    """Rasterise to an (H, W, 3) uint8 image, averaging colour per cell.

    The top slice is discarded first. Looking down at a room otherwise shows the
    ceiling and nothing else, which is a correct rendering of a useless view.
    """
    up = points[:, transform.up_axis]
    if 0.0 < drop_ceiling < 1.0 and len(up):
        keep = up <= np.quantile(up, 1.0 - drop_ceiling)
        if keep.sum() > 100:
            points, colors = points[keep], colors[keep]

    px = transform.world_to_pixel(points)
    u = np.clip(px[:, 0].astype(np.int32), 0, transform.width - 1)
    v = np.clip(px[:, 1].astype(np.int32), 0, transform.height - 1)
    flat = v * transform.width + u

    cells = transform.width * transform.height
    counts = np.bincount(flat, minlength=cells).astype(np.float32)

    image = np.zeros((cells, 3), dtype=np.float32)
    for c in range(3):
        image[:, c] = np.bincount(flat, weights=colors[:, c].astype(np.float32),
                                  minlength=cells)

    nonempty = counts > 0
    image[nonempty] /= counts[nonempty, None]

    # Empty cells stay dark rather than black, so the room's footprint reads as
    # a shape against a background instead of merging into it.
    out = np.full((cells, 3), 18, dtype=np.uint8)
    out[nonempty] = np.clip(image[nonempty], 0, 255).astype(np.uint8)
    return out.reshape(transform.height, transform.width, 3)


def object_footprints(objects: list[dict], transform: PlanTransform) -> list[dict]:
    """Project each object's OBB corners onto the plan as a polygon + centre."""
    out = []
    for obj in objects:
        obb = obj.get("obb") or {}
        try:
            center = np.asarray(obb["center"], dtype=float)
            extent = np.asarray(obb["extent"], dtype=float)
            R = np.asarray(obb["R"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue

        signs = np.array(
            [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]
        )
        corners = (signs * extent[None, :]) @ R.T + center[None, :]

        poly = transform.world_to_pixel(corners)
        centroid_px = transform.world_to_pixel(np.asarray(obj["centroid"], dtype=float))[0]

        out.append(
            {
                "id": obj.get("id"),
                "label": obj.get("label"),
                "hull": _convex_hull(poly).tolist(),
                "centroid_px": [round(float(centroid_px[0]), 2),
                                round(float(centroid_px[1]), 2)],
            }
        )
    return out


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone chain. Eight projected corners do not justify pulling in scipy."""
    pts = np.unique(np.round(points, 4), axis=0)
    if len(pts) < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    return np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
