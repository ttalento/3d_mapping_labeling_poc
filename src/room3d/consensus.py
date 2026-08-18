"""Do the other cameras agree that this point is the object?

A 2D detection box is axis-aligned; the object inside it is not, and is not
rectangular either. So the box always contains things that are not the object --
the sofa standing in front of the cabinet, a strip of floor, the wall behind. No
amount of work on the *box* fixes this, because the information is not in that
frame.

It is in the other frames. Clutter sits at a different depth from the object, so
under parallax it projects outside the box seen from another angle, while the
object projects inside every one of them. Counting those agreements is the whole
method, and it needs nothing that is not already on disk.

Two rules make the count mean something:

- A frame that could not *see* a point does not vote on it. Occlusion is not
  disagreement (see `camera.visible_in_frame`).
- A point does not vote for itself. A point harvested from frame 6's box is
  trivially inside frame 6's box; counting that as verification inflates every
  score by exactly one frame, so a two-view object looks half-confirmed when
  nothing has confirmed it.

`min_vote` is a *fraction* of the frames that could see the point, never a veto.
Requiring unanimity was the first thing tried and it destroyed a speaker seen by
two cameras.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .camera import visible_in_frame


@dataclass(frozen=True)
class View:
    """One 2D box in one frame -- the cached artifact this all rests on.

    The provenance fields are carried, not used, by the geometry here. They are
    what lets a query say which stored observations and which existing objects a
    result subsumes.
    """

    frame_idx: int
    box_px: tuple[int, int, int, int]        # x0, y0, x1, y1; x1/y1 exclusive
    label: str = ""
    vlm_confidence: float = 1.0
    observation_id: int | None = None
    object_id: str | None = None


def vote(
    points: np.ndarray,
    views: Sequence[View],
    recon,
    *,
    occlusion_tol: float = 0.10,
    source: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per point: how many views agree, and how many were entitled to an opinion.

    Returns `(agree, testable)`. Both counts matter -- a point with 2 of 2 is
    better evidenced than one with 2 of 12, and dividing too early throws that
    away.

    `source[i]` is the frame point `i` was harvested from; that frame abstains.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    agree = np.zeros(len(pts))
    testable = np.zeros(len(pts))
    if len(pts) == 0:
        return agree, testable

    src = None if source is None else np.asarray(source).reshape(-1)

    for view in views:
        uv, seen = visible_in_frame(pts, recon, view.frame_idx, tol=occlusion_tol)
        if src is not None:
            seen = seen & (src != view.frame_idx)

        x0, y0, x1, y1 = view.box_px
        inside = (
            seen
            & (uv[:, 0] >= x0) & (uv[:, 0] < x1)
            & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
        )
        agree += inside
        testable += seen

    return agree, testable


def consistent(
    points: np.ndarray,
    views: Sequence[View],
    recon,
    *,
    min_vote: float = 0.6,
    occlusion_tol: float = 0.10,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """Which points enough of the entitled views agree on.

    A point no view could judge is never consistent. Absence of evidence is not
    evidence, and treating it as agreement would keep exactly the points that
    nothing has checked.
    """
    agree, testable = vote(
        points, views, recon, occlusion_tol=occlusion_tol, source=source
    )
    judged = testable > 0
    fraction = np.divide(agree, np.maximum(testable, 1.0))
    return judged & (fraction >= min_vote - 1e-9)


def candidate_points(
    views: Sequence[View], recon
) -> tuple[np.ndarray, np.ndarray]:
    """Every confident 3D point inside any of the boxes, plus where each came from.

    The union, not the intersection: a partial view contributes the part of the
    object it can see, and the vote below is what removes what it should not
    have contributed.
    """
    chunks: list[np.ndarray] = []
    sources: list[np.ndarray] = []

    for view in views:
        x0, y0, x1, y1 = view.box_px
        mask = np.zeros(recon.image_hw, dtype=bool)
        mask[y0:y1, x0:x1] = True
        mask &= recon.conf_mask[view.frame_idx]

        pts = np.asarray(recon.pts3d[view.frame_idx], dtype=np.float64)[mask]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts):
            chunks.append(pts)
            sources.append(np.full(len(pts), view.frame_idx, dtype=np.int64))

    if not chunks:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.vstack(chunks), np.concatenate(sources)


def largest_component(
    points: np.ndarray, *, voxel: float = 0.04, max_cells: int = 20_000_000
) -> np.ndarray:
    """Keep only the biggest spatially connected blob.

    Carving removes clutter the other views disagree about. What it cannot remove
    is clutter they all agree about -- a chair standing in front of a cabinet in
    every frame. Often that clutter is nonetheless *spatially separate* from the
    object, and then this finds it: voxelise, label 26-connected components, keep
    the largest.

    It does nothing when the clutter physically touches the object, because then
    they genuinely are one connected mass of points. That limit is real and is
    documented in the spec rather than papered over.

    `max_cells` guards the grid: one far-flung outlier with a fine `voxel` would
    otherwise allocate gigabytes, so the resolution is coarsened until the grid
    fits.
    """
    from scipy import ndimage

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return np.ones(len(pts), dtype=bool)

    span = pts.max(axis=0) - pts.min(axis=0)
    voxel = float(max(voxel, 1e-9))
    while float(np.prod(np.floor(span / voxel) + 3.0)) > max_cells:
        voxel *= 2.0

    idx = np.floor((pts - pts.min(axis=0)) / voxel).astype(np.int64) + 1
    grid = np.zeros(tuple(idx.max(axis=0) + 2), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=int))
    if n <= 1:
        return np.ones(len(pts), dtype=bool)

    per_point = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    counts = np.bincount(per_point, minlength=n + 1)
    counts[0] = 0
    return per_point == int(np.argmax(counts))


@dataclass
class CarveResult:
    """What survived, where each surviving point came from, and how it went.

    `source` is carried out of the carve because the only honest way to report
    how strongly a point set is agreed on is with leave-one-out, and that needs
    each point's originating frame. Without it every caller would have to either
    recompute the carve or quote a vote fraction inflated by self-agreement.
    """

    points: np.ndarray                  # (K, 3)
    source: np.ndarray                  # (K,) originating frame per point
    stats: dict


def carve(
    views: Sequence[View],
    recon,
    *,
    min_vote: float = 0.6,
    occlusion_tol: float = 0.10,
    keep_largest: bool = True,
    voxel: float = 0.04,
) -> CarveResult:
    """The points that are actually the object these views describe.

    With fewer than two views there is nothing to cross-check against, so the
    candidates are returned untouched and `mean_vote` is 0.0 -- nothing verified
    them. Returning an empty set there would read as "the object is not present"
    when the truth is "nothing could verify it"; callers report the view count so
    the difference stays visible.
    """
    points, source = candidate_points(views, recon)
    stats = {
        "min_vote": min_vote,
        "n_candidates": int(len(points)),
        "n_kept": int(len(points)),
        "kept_frac": 1.0,
        "mean_vote": 0.0,
    }

    if len(points) == 0 or len(views) < 2:
        return CarveResult(points, source, stats)

    keep = consistent(
        points, views, recon,
        min_vote=min_vote, occlusion_tol=occlusion_tol, source=source,
    )
    kept, kept_source = points[keep], source[keep]

    if keep_largest and len(kept) >= 2:
        biggest = largest_component(kept, voxel=voxel)
        kept, kept_source = kept[biggest], kept_source[biggest]

    stats["n_kept"] = int(len(kept))
    stats["kept_frac"] = round(float(len(kept) / max(len(points), 1)), 4)
    stats["mean_vote"] = round(_mean_vote(kept, kept_source, views, recon, occlusion_tol), 4)
    return CarveResult(kept, kept_source, stats)


def _mean_vote(points, source, views, recon, occlusion_tol: float) -> float:
    """Average agreement across the points that survived, leave-one-out."""
    if len(points) == 0:
        return 0.0
    agree, testable = vote(
        points, views, recon, occlusion_tol=occlusion_tol, source=source
    )
    judged = testable > 0
    if not judged.any():
        return 0.0
    return float((agree[judged] / testable[judged]).mean())
