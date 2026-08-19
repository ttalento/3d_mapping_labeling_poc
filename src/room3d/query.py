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

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .artifacts import load_frames_npz
from .config import QueryConfig
from .consensus import View, carve
from .fusion import default_label_compatible, normalize_label
from .level import estimate_up
from .projection import OrientedBox, fit_gravity_aligned_box, snap_to_floor

_ARTICLES = ("the ", "a ", "an ")

# Matches LabelConfig.min_level_confidence. Below this the levelling estimate is
# not worth acting on, and a box levelled against a wrong gravity vector is worse
# than an unlevelled one.
MIN_LEVEL_CONFIDENCE = 0.25


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


def _box_area(box_px: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box_px
    return max(x1 - x0, 0) * max(y1 - y0, 0)


def group_instances(
    views: Sequence[View],
    recon,
    *,
    min_agreement: float = 0.3,
    occlusion_tol: float = 0.10,
    max_points: int = 2000,
) -> list[list[View]]:
    """Split the matched detections into distinct physical objects.

    "chair" matches every chair in the room. Which detections are the *same*
    chair is decided by `agreement_between` -- the same cross-view vote used to
    carve -- rather than by centroid distance. That matters because centroid
    distance is only a proxy for the question, and it is the proxy that puts one
    couch in the object list twice.

    Greedy, largest box first, so well-supported detections seed the groups and
    marginal ones attach to them. `fusion.cluster_observations` orders itself the
    same way and for the same reason.
    """
    from .consensus import agreement_between

    groups: list[list[View]] = []

    for view in sorted(views, key=lambda v: -_box_area(v.box_px)):
        best: list[View] | None = None
        best_score = min_agreement

        for group in groups:
            score = agreement_between(
                [view], group, recon,
                occlusion_tol=occlusion_tol, max_points=max_points,
            )
            if score >= best_score:
                best, best_score = group, score

        if best is None:
            groups.append([view])
        else:
            best.append(view)

    return sorted(groups, key=lambda g: -len(g))


@dataclass
class QueryMatch:
    """One physical object the query found, and the evidence behind it."""

    label: str
    obb: OrientedBox | None
    score: float
    views: list[View]
    n_points: int
    vote_stats: dict
    absorbed_object_ids: list[str] = field(default_factory=list)
    supported: bool = True

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "obb": self.obb.as_dict() if self.obb is not None else None,
            "score": round(float(self.score), 4),
            "n_views": len(self.views),
            "n_points": int(self.n_points),
            "frames": sorted({v.frame_idx for v in self.views}),
            "boxes": {str(v.frame_idx): list(v.box_px) for v in self.views},
            "vote_stats": self.vote_stats,
            "absorbed_object_ids": list(self.absorbed_object_ids),
            "supported": bool(self.supported),
        }


@dataclass
class QueryResult:
    phrase: str
    matches: list[QueryMatch]
    source: str                       # "cache" | "vlm" | "none"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "phrase": self.phrase,
            "source": self.source,
            "matches": [m.as_dict() for m in self.matches],
            "notes": list(self.notes),
        }


def query_room(
    room_dir: str | Path,
    phrase: str,
    *,
    config: QueryConfig | None = None,
    config_overrides: dict | None = None,
    detector=None,
    force: bool = False,
    label_compatible: Callable[[str, str], bool] = default_label_compatible,
    verbose: bool = True,
) -> QueryResult:
    """Find `phrase` in a labelled room and return its box, best match first.

    Reads only. Promoting a match into `objects.json` is `commit_match`'s job,
    and keeping them separate is what makes a vague query free rather than
    destructive.
    """
    room_dir = Path(room_dir)
    npz = room_dir / "frames.npz"
    obs_path = room_dir / "observations.json"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found; reconstruct this room first")
    if not obs_path.exists():
        raise FileNotFoundError(f"{obs_path} not found; label this room first")

    config = config or QueryConfig()
    if config_overrides:
        config = replace(config, **config_overrides)

    recon = load_frames_npz(npz)
    doc = json.loads(obs_path.read_text())
    notes: list[str] = []

    views: list[View] = []
    source = "none"
    if not force:
        views = cached_views(doc, phrase, label_compatible=label_compatible)
        if views:
            source = "cache"

    if not views:
        if detector is None:
            notes.append(
                "nothing cached matches that phrase, and no detector is "
                "available to look for it -- this is not the same as the object "
                "being absent from the room"
            )
            return QueryResult(phrase, [], "none", notes)
        views = _vlm_views(recon, doc, phrase, detector, notes)
        source = "vlm"

    up, floor_height = _scene_frame(recon, notes)

    matches = [
        _build_match(group, recon, up, floor_height, config)
        for group in group_instances(
            views, recon,
            min_agreement=config.min_instance_agreement,
            occlusion_tol=config.occlusion_tol,
            max_points=config.max_agreement_points,
        )
    ]
    matches.sort(key=lambda m: -m.score)

    if verbose:
        print(f"[query] {phrase!r} -> {len(matches)} match(es) from {source}")
        for i, m in enumerate(matches, 1):
            size = "unsupported" if m.obb is None else np.round(m.obb.extent, 3).tolist()
            print(f"[query]   {i}. {m.label:<16} score={m.score:.2f} "
                  f"views={len(m.views)} extent={size}")
        for note in notes:
            print(f"[query] note: {note}")

    return QueryResult(phrase, matches, source, notes)


def _scene_frame(recon, notes: list[str]):
    """`(up, floor_height)` for the room, or `(None, None)` if untrustworthy."""
    estimate = estimate_up(recon.pts3d[recon.conf_mask], recon.poses)
    if estimate.confidence < MIN_LEVEL_CONFIDENCE:
        notes.append(
            f"up estimate too weak to trust ({estimate.confidence:.2f}); "
            "the box will not be levelled"
        )
        return None, None
    return estimate.up, float(estimate.floor_offset)


def _build_match(group, recon, up, floor_height, config: QueryConfig) -> QueryMatch:
    """Carve one instance's points and wrap a box around what survives."""
    result = carve(
        group, recon,
        min_vote=config.min_vote,
        occlusion_tol=config.occlusion_tol,
        keep_largest=config.keep_largest_component,
        voxel=config.component_voxel,
    )
    kept, stats = result.points, result.stats

    absorbed = sorted({v.object_id for v in group if v.object_id})
    label = _dominant_label(group)

    if len(kept) < 3:
        # Found in 2D and unsupportable in 3D. Reporting it as a match with
        # supported=False says that; returning nothing would say "not present".
        return QueryMatch(label, None, 0.0, list(group), 0, stats, absorbed, False)

    if up is None:
        from .projection import fit_oriented_box

        obb = fit_oriented_box(kept)
    else:
        obb = fit_gravity_aligned_box(kept, up)
        if floor_height is not None:
            obb = snap_to_floor(obb, floor_height, up)

    return QueryMatch(
        label=label,
        obb=obb,
        score=_score(group, stats["mean_vote"]),
        views=list(group),
        n_points=int(len(kept)),
        vote_stats=stats,
        absorbed_object_ids=absorbed,
        supported=True,
    )


def _dominant_label(group: Sequence[View]) -> str:
    counts: dict[str, int] = {}
    for view in group:
        name = normalize_label(view.label)
        counts[name] = counts.get(name, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))[0] if counts else ""


def _score(group: Sequence[View], mean_vote: float) -> float:
    """A within-query ranking key. Not a probability, not comparable across queries.

    Geometric mean of three independent signals -- how many views support the
    match, how strongly the candidate points are agreed on (`carve`'s
    `stats["mean_vote"]`, averaged over every candidate rather than only the
    survivors of the `min_vote` cut -- otherwise carving harder would report
    higher confidence, not lower), and how sure the VLM was. `_cluster_confidence`
    in `fusion.py` combines its terms the same way and for the same reason: a
    match that fails any one of them must not be rescued by the other two.
    """
    view_score = 1.0 - np.exp(-len(group) / 2.0)
    vlm_score = float(np.mean([v.vlm_confidence for v in group])) if group else 0.0
    terms = (max(view_score, 1e-6), max(mean_vote, 1e-6), max(vlm_score, 1e-6))
    return float(np.clip(np.prod(terms) ** (1 / 3), 0.0, 1.0))


def _vlm_views(recon, doc, phrase, detector, notes: list[str]) -> list[View]:
    """Ask the detector for `phrase` in every frame the room was labelled on.

    Those frames were already chosen for coverage by `select_covering_frames`;
    re-deriving the selection here would spend calls to arrive at the same list.

    One frame failing must not lose the other eleven. Gemini's free tier returns
    429 often enough that an all-or-nothing query would frequently return
    nothing at all, which is indistinguishable from the object being absent.
    """
    from .projection import descale_box

    frames = [int(i) for i in doc.get("frames_labeled", [])]
    frames = [i for i in frames if i < recon.n_frames] or list(range(recon.n_frames))

    height, width = recon.image_hw
    views: list[View] = []
    failures = 0

    for frame_idx in frames:
        try:
            detections = detector.locate(recon.images[frame_idx], phrase)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not lose the rest
            failures += 1
            notes.append(f"frame {frame_idx} failed: {exc}")
            continue

        for det in detections:
            views.append(
                View(
                    frame_idx=frame_idx,
                    box_px=descale_box(det.box_2d, height, width),
                    label=det.label or phrase,
                    vlm_confidence=float(det.confidence),
                )
            )

    if failures:
        notes.append(f"{failures} of {len(frames)} frames failed to be searched")
    if not views:
        notes.append(f"the detector found nothing matching {phrase!r} in any frame")
    return views


def filter_by_certainty(
    matches: Sequence[QueryMatch],
    *,
    min_views: int,
    min_mean_vote: float,
) -> tuple[list[QueryMatch], list[QueryMatch]]:
    """Split matches into those we can place and those we cannot: `(kept, dropped)`.

    Both halves are returned because nothing may vanish silently. A filter that
    quietly halves the object list is indistinguishable from a bug, and the
    request was to lose uncertain labels -- not to lose the knowledge that they
    were found.

    A single-view match fails this gate by construction: cross-view agreement is
    the only evidence here that a box is where it claims to be, and one view has
    nothing to be checked against.
    """
    kept: list[QueryMatch] = []
    dropped: list[QueryMatch] = []
    for m in matches:
        certain = (
            m.supported
            and len(m.views) >= min_views
            and float(m.vote_stats.get("mean_vote", 0.0)) >= min_mean_vote
        )
        (kept if certain else dropped).append(m)
    return kept, dropped


def commit_match(
    room_dir: str | Path,
    match: QueryMatch,
    *,
    config: QueryConfig | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Write a match into `objects.json`, replacing what it subsumes.

    This is the step that fixes duplicates. A query that finds one couch where
    the object list holds two removes both and writes one, on the strength of the
    cross-view evidence rather than a threshold that happened to work.

    Destructive, so it backs up first: a vague phrase must never cost a labelled
    room. Same `.prev.json` convention as `refit.py`.

    Refuses a match below the certainty gate unless `force=True`: committing
    writes a position into `objects.json` that later code will treat as fact,
    and an uncertain box acted on is worse than one never written. The gate
    uses `config` -- default `QueryConfig()` if the caller has none -- rather
    than a second, independently-defaulted `QueryConfig()` of its own, because
    a caller that already decided a match is committable (against, say, a
    `--config` override loosening `min_views`) must not have that decision
    re-litigated here against different numbers and refused.
    """
    from .fusion import ObjectRecord

    if not match.supported or match.obb is None:
        raise ValueError(
            "this match is unsupported -- it was found in 2D but no 3D points "
            "survived carving, so there is no box to commit"
        )

    if not force:
        config = config or QueryConfig()
        if not filter_by_certainty(
            [match], min_views=config.min_views, min_mean_vote=config.min_mean_vote
        )[0]:
            raise ValueError(
                f"this match is below the certainty gate ({len(match.views)} view(s), "
                f"mean vote {match.vote_stats.get('mean_vote', 0.0):.2f}) -- writing it "
                f"into objects.json would record a position nothing verified. "
                f"Pass force=True to commit it anyway."
            )

    room_dir = Path(room_dir)
    path = room_dir / "objects.json"
    doc = json.loads(path.read_text()) if path.exists() else {"objects": []}
    objects = list(doc.get("objects", []))

    absorbed = set(match.absorbed_object_ids)
    kept = [o for o in objects if o.get("id") not in absorbed]
    removed = [o["id"] for o in objects if o.get("id") in absorbed]

    # Reuse an absorbed id where there is one: the new object *is* the old one,
    # better measured, and stable ids keep any external references working.
    if match.absorbed_object_ids:
        object_id = sorted(match.absorbed_object_ids)[0]
    else:
        used = {o.get("id", "") for o in kept}
        n = 0
        while f"obj_{n:03d}" in used:
            n += 1
        object_id = f"obj_{n:03d}"

    labels = {normalize_label(v.label) for v in match.views}
    record = ObjectRecord(
        id=object_id,
        label=match.label,
        aliases=sorted(labels - {normalize_label(match.label)}),
        centroid=np.asarray(match.obb.center, dtype=float),
        obb=match.obb,
        confidence=match.score,
        n_observations=len(match.views),
        seen_in=sorted({v.frame_idx for v in match.views}),
        observation_ids=sorted(
            v.observation_id for v in match.views if v.observation_id is not None
        ),
    )

    # Metadata (units, scale_verified, ...) is carried over, not re-derived:
    # whether the metric scale was checked against a tape measure is a fact
    # about the capture, and nothing here re-measures it.
    path.with_suffix(".prev.json").write_text(json.dumps(doc, indent=2))
    doc["objects"] = kept + [record.as_dict()]
    path.write_text(json.dumps(doc, indent=2))

    _repoint_absorbed_observations(room_dir, absorbed, object_id)

    if verbose:
        print(f"[query] committed {object_id} ({match.label}); "
              f"removed {len(removed)} absorbed object(s)")

    return {"object_id": object_id, "removed": removed, "n_objects": len(doc["objects"])}


def _repoint_absorbed_observations(
    room_dir: Path, absorbed: set, object_id: str
) -> None:
    """Rewrite every absorbed observation's `object_id` to the committed id.

    `commit_match` only ever rewrites `objects.json`; without this, an
    observation that used to belong to an absorbed object keeps pointing at an
    id that no longer exists in `objects.json`. The webapp associates
    observations to objects by that field
    (`o.object_id === S.selected` in `app.js`), so a dangling link makes a
    merged object display only a subset of the frames it was built from.

    Same `.prev.json` backup discipline as `objects.json`: destructive, so it
    backs up first.
    """
    if not absorbed:
        return

    obs_path = room_dir / "observations.json"
    if not obs_path.exists():
        return

    original = obs_path.read_text()
    doc = json.loads(original)
    observations = doc.get("observations", [])

    changed = False
    for item in observations:
        if item.get("object_id") in absorbed:
            item["object_id"] = object_id
            changed = True

    if not changed:
        return

    obs_path.with_suffix(".prev.json").write_text(original)
    obs_path.write_text(json.dumps(doc, indent=2))
