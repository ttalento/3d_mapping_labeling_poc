"""Name an object, get its box.

The pipeline's job is to label everything; this is for when you want *one*
thing and want it right. It changes what the geometry is anchored to. Fusion
must guess which observations belong together from centroid distance, and when
it guesses wrong a couch appears twice. A query names the object, so identity is
given rather than inferred, and the machinery is free to spend its effort on
where the object actually is.

Nothing here recomputes the reconstruction. The 2D detections are already on
disk and, per the user, they are good; what was missing was a way to combine
them that respects the fact that a box is not an object.

Order of operations:

    phrase -> views -> instances -> carve -> box -> ranked matches

Cached detections are tried first because they are free. The VLM is called only
when the phrase matches nothing, or when explicitly forced.
"""

from __future__ import annotations

from collections.abc import Callable

from .consensus import View
from .fusion import default_label_compatible, normalize_label

_ARTICLES = ("the ", "a ", "an ")


def normalize_phrase(phrase: str) -> str:
    """Lowercase, collapse whitespace, and strip one leading article.

    Only a *leading* article. "the couch" is a request for the couch; "the couch
    by the window" is a request that cached labels cannot evaluate, and stripping
    the inner "the" would not make it evaluable -- it would only make the phrase
    look like it had been understood.
    """
    text = normalize_label(phrase)
    for article in _ARTICLES:
        if text.startswith(article):
            return text[len(article):].strip()
    return text


def cached_views(
    observations_doc: dict,
    phrase: str,
    *,
    label_compatible: Callable[[str, str], bool] = default_label_compatible,
) -> list[View]:
    """Every stored detection whose label matches `phrase`.

    Returns detections belonging to several different objects when the phrase is
    ambiguous ("chair"). Separating them is a later stage's job -- doing it here
    would mean guessing which chair was meant before anything has looked at the
    geometry.

    An empty list means the cache cannot answer, which is the signal to fall
    through to the VLM. That is also how a qualified phrase is handled: it
    matches no label, so it becomes a miss rather than a wrong answer.
    """
    target = normalize_phrase(phrase)
    views: list[View] = []

    for i, item in enumerate(observations_doc.get("observations", [])):
        box = item.get("box_px")
        if not box:
            continue
        if not label_compatible(str(item.get("label", "")), target):
            continue
        views.append(
            View(
                frame_idx=int(item["frame_idx"]),
                box_px=tuple(int(v) for v in box),
                label=str(item.get("label", "")),
                vlm_confidence=float(item.get("vlm_confidence", 0.5)),
                observation_id=int(item.get("id", i)),
                object_id=item.get("object_id"),
            )
        )
    return views
