"""Merge per-frame observations into unique objects.

One chair seen in six frames must become one object; two different chairs must
not become one. Those failure modes pull in opposite directions, so no single
distance threshold wins. The structure here is:

    deterministic geometry proposes  ->  the LLM only adjudicates naming

Geometry decides *what could be the same thing* using a size-adaptive merge
radius (a sofa and a mug cannot share a threshold) plus an overlap gate. The
language model is consulted for exactly one question it is actually good at:
whether "monitor", "computer screen" and "display" name the same object.

`label_compatible` and `canonicalize` are injected so this module stays pure and
testable; the CrewAI layer passes LLM-backed versions, the tests pass trivial
ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .projection import (
    Observation,
    OrientedBox,
    fit_gravity_aligned_box,
    snap_to_floor,
    subsample,
)

# Deliberately small: this is a fallback for when the LLM adjudicator is absent
# (unit tests, --no-llm-merge). The real synonym handling is the agent's job.
# First entry of each group is the canonical name.
_DEFAULT_SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("monitor", "computer screen", "display", "screen"),
    ("sofa", "couch", "settee"),
    ("desk", "table", "computer desk"),
    ("bin", "rubbish bin", "trash can", "wastebasket"),
)


@dataclass
class ObjectRecord:
    id: str
    label: str
    aliases: list[str]
    centroid: np.ndarray
    obb: OrientedBox
    confidence: float
    n_observations: int
    seen_in: list[int] = field(default_factory=list)
    # Indices into the observation list passed to cluster_observations. This is
    # what lets a viewer draw the object back onto the exact 2D boxes it came
    # from; `seen_in` collapses to unique frames and cannot do that, since one
    # frame can contribute several observations to the same object.
    observation_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "aliases": self.aliases,
            "centroid": [round(float(v), 4) for v in self.centroid],
            "obb": self.obb.as_dict(),
            "confidence": round(float(self.confidence), 4),
            "n_observations": self.n_observations,
            "seen_in": self.seen_in,
            "observation_ids": self.observation_ids,
        }


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").split())


def default_label_compatible(a: str, b: str) -> bool:
    """Deterministic fallback: exact match after normalisation, or a known synonym."""
    na, nb = normalize_label(a), normalize_label(b)
    if na == nb:
        return True
    return any(na in group and nb in group for group in _DEFAULT_SYNONYMS)


def default_canonicalize(labels: Sequence[str]) -> str:
    """Most frequent label wins.

    Ties go to the synonym group's primary name if one applies -- the table
    already encodes that "monitor" is the name to prefer over "display" -- and
    otherwise to the shorter, then alphabetical, label.
    """
    counts: dict[str, int] = {}
    for lab in labels:
        norm = normalize_label(lab)
        counts[norm] = counts.get(norm, 0) + 1

    def rank(item: tuple[str, int]) -> tuple:
        name, count = item
        primary = next(
            (0 for group in _DEFAULT_SYNONYMS if name == group[0]),
            1,
        )
        return (-count, primary, len(name), name)

    return min(counts.items(), key=rank)[0]


def aabb_of(obb: OrientedBox) -> tuple[np.ndarray, np.ndarray]:
    """World-axis-aligned envelope of an oriented box.

    A true OBB-vs-OBB intersection volume needs convex polyhedron clipping. This
    gate only has to be roughly right — it is a cheap veto in front of the
    centroid-distance test, not the merge decision itself — so the axis-aligned
    envelope is the right amount of machinery.
    """
    corners = np.array(
        [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]
    ) * obb.extent[None, :]
    world = corners @ obb.R.T + obb.center[None, :]
    return world.min(axis=0), world.max(axis=0)


def aabb_iou(a: OrientedBox, b: OrientedBox) -> float:
    lo_a, hi_a = aabb_of(a)
    lo_b, hi_b = aabb_of(b)

    lo = np.maximum(lo_a, lo_b)
    hi = np.minimum(hi_a, hi_b)
    overlap = np.prod(np.clip(hi - lo, 0.0, None))
    if overlap <= 0.0:
        return 0.0

    vol_a = float(np.prod(np.clip(hi_a - lo_a, 1e-9, None)))
    vol_b = float(np.prod(np.clip(hi_b - lo_b, 1e-9, None)))
    union = vol_a + vol_b - overlap
    return float(overlap / union) if union > 0 else 0.0


def _merge_radius(obs: Sequence[Observation], floor: float, scale: float) -> float:
    """Scale the merge threshold to object size, never below `floor`."""
    diags = [o.obb.diagonal for o in obs if o.obb.diagonal > 0]
    mean_diag = float(np.mean(diags)) if diags else 0.0
    return max(floor, scale * mean_diag)


def _cluster_confidence(obs: Sequence[Observation], radius: float) -> float:
    """Blend how often it was seen, how sure the VLM was, and how tightly the
    observations agree in 3D. An object seen once, hesitantly, with scattered
    geometry should rank below one seen five times with a tight cluster."""
    n = len(obs)
    obs_score = 1.0 - np.exp(-n / 2.0)
    vlm_score = float(np.mean([o.vlm_confidence for o in obs]))

    centroids = np.stack([o.centroid for o in obs])
    spread = float(np.mean(np.linalg.norm(centroids - centroids.mean(axis=0), axis=1)))
    tight_score = float(np.exp(-spread / max(radius, 1e-6)))

    return float(
        np.clip(obs_score**0.35 * max(vlm_score, 1e-6) ** 0.40 * tight_score**0.25, 0.0, 1.0)
    )


def cluster_observations(
    observations: Sequence[Observation],
    *,
    label_compatible: Callable[[str, str], bool] = default_label_compatible,
    canonicalize: Callable[[Sequence[str]], str] = default_canonicalize,
    radius_floor: float = 0.30,
    radius_scale: float = 0.50,
    min_obb_iou: float = 0.10,
    up: np.ndarray | None = None,
    floor_height: float | None = None,
    floor_snap_threshold: float = 0.15,
    obb_percentile: float = 1.0,
    max_pooled_points: int = 20_000,
) -> list[ObjectRecord]:
    """Greedy agglomerative clustering of observations into objects.

    An observation joins a cluster when geometry *and* label both agree:
      - centroid within a size-adaptive radius of the *nearest member* of the
        cluster, and
      - either the boxes overlap (IoU >= min_obb_iou) or that distance is
        comfortably inside the radius (which rescues thin objects whose
        axis-aligned envelopes barely intersect), and
      - the label is compatible with the cluster's existing labels.

    Nearest member, not the mean of the cluster's centroids. A sofa filmed by
    walking past it produces views that each overlap their neighbours and not the
    far end; once two of them merge, their mean sits between them, and every
    later view is measured from a place no observation ever was. The far half of
    the sofa then splits off as a second sofa. What the gate is actually asking
    is whether this view coincides with something already in the cluster, and
    that is a question about members.

    Observations are processed most-supported first, so well-grounded detections
    seed the clusters and marginal ones attach to them rather than the reverse.

    `up` and `floor_height` govern the *final* box only, not the clustering.
    Given them, each object's box is refit to the pooled points of every
    observation in its cluster, level with gravity and resting on the floor.
    """
    # Carry the caller's index alongside each observation so the finished record
    # can point back at the exact observations -- and therefore the exact 2D
    # boxes -- that produced it.
    indexed = sorted(
        enumerate(observations), key=lambda io: (-io[1].n_points, io[1].frame_idx)
    )
    clusters: list[list[tuple[int, Observation]]] = []

    for idx, obs in indexed:
        target = None
        best_distance = np.inf

        for cluster in clusters:
            if not any(label_compatible(obs.label, c.label) for _, c in cluster):
                continue

            members = [c for _, c in cluster]
            radius = _merge_radius(members + [obs], radius_floor, radius_scale)
            distance = min(
                float(np.linalg.norm(obs.centroid - c.centroid)) for c in members
            )
            if distance > radius:
                continue

            overlaps = any(aabb_iou(obs.obb, c.obb) >= min_obb_iou for c in members)
            if not (overlaps or distance <= 0.5 * radius):
                continue

            if distance < best_distance:
                best_distance, target = distance, cluster

        if target is None:
            clusters.append([(idx, obs)])
        else:
            target.append((idx, obs))

    return [
        _finalise(
            f"obj_{i:03d}", c, canonicalize, radius_floor, radius_scale,
            up=up,
            floor_height=floor_height,
            floor_snap_threshold=floor_snap_threshold,
            obb_percentile=obb_percentile,
            max_pooled_points=max_pooled_points,
        )
        for i, c in enumerate(sorted(clusters, key=lambda c: -len(c)))
    ]


def _fuse_box(
    cluster: Sequence[Observation],
    up: np.ndarray | None,
    floor_height: float | None,
    floor_snap_threshold: float,
    obb_percentile: float,
    max_pooled_points: int,
) -> OrientedBox:
    """One box for the whole cluster, fit to everything every frame saw.

    The only honest way to build this is from points. Averaging per-frame boxes
    cannot work: their extents are expressed along *different* axes, so an
    element-wise combination of two of them describes no box at all. Where the
    points are unavailable -- re-fusion from `observations.json`, or a synthetic
    test -- the fallback is a box that genuinely existed rather than a mixture.
    """
    pooled = [o.points for o in cluster if o.points is not None and len(o.points)]
    if not pooled or up is None:
        return max(cluster, key=lambda o: o.obb.diagonal).obb

    points = subsample(np.vstack(pooled), max_pooled_points)
    obb = fit_gravity_aligned_box(points, up, percentile=obb_percentile)
    if floor_height is not None:
        obb = snap_to_floor(obb, floor_height, up, threshold=floor_snap_threshold)
    return obb


def _finalise(
    obj_id: str,
    indexed_cluster: Sequence[tuple[int, Observation]],
    canonicalize: Callable[[Sequence[str]], str],
    radius_floor: float,
    radius_scale: float,
    *,
    up: np.ndarray | None = None,
    floor_height: float | None = None,
    floor_snap_threshold: float = 0.15,
    obb_percentile: float = 1.0,
    max_pooled_points: int = 20_000,
) -> ObjectRecord:
    ids = [i for i, _ in indexed_cluster]
    cluster = [o for _, o in indexed_cluster]

    labels = [o.label for o in cluster]
    canonical = canonicalize(labels)
    aliases = sorted({normalize_label(l) for l in labels} - {normalize_label(canonical)})

    centroids = np.stack([o.centroid for o in cluster])
    radius = _merge_radius(cluster, radius_floor, radius_scale)
    obb = _fuse_box(
        cluster, up, floor_height, floor_snap_threshold, obb_percentile, max_pooled_points
    )

    return ObjectRecord(
        id=obj_id,
        label=canonical,
        aliases=aliases,
        centroid=np.median(centroids, axis=0),
        obb=obb,
        confidence=_cluster_confidence(cluster, radius),
        n_observations=len(cluster),
        seen_in=sorted({int(o.frame_idx) for o in cluster}),
        observation_ids=sorted(int(i) for i in ids),
    )
