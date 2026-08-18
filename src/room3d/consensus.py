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
