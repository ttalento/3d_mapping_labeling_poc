"""Re-cluster cached observations with different thresholds.

The observations are the expensive part of a run -- one VLM call per frame.
Clustering them is pure arithmetic over a few dozen records, so re-running it
with a different merge radius costs milliseconds and no API calls. That makes
the fusion thresholds explorable rather than a thing you guess once in a YAML
file and re-run the whole pipeline to test.

This deliberately calls `fusion.cluster_observations` as-is rather than
reimplementing it. If the two ever diverge, the app would be tuning against
something the pipeline does not actually do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..artifacts import load_observation_points
from ..fusion import ObjectRecord, cluster_observations
from ..level import UP_VECTOR, load_level_record
from ..projection import Observation, OrientedBox


@dataclass
class ObservationSet:
    room: str
    image_hw: tuple[int, int]
    frames_labeled: list[int]
    observations: list[Observation]
    boxes: list[tuple[int, int, int, int] | None]

    @property
    def n(self) -> int:
        return len(self.observations)


def _orientedbox_from_dict(d: dict) -> OrientedBox:
    return OrientedBox(
        center=np.asarray(d["center"], dtype=float),
        extent=np.asarray(d["extent"], dtype=float),
        R=np.asarray(d["R"], dtype=float),
    )


def load_observations(path: str | Path) -> ObservationSet:
    """Read observations.json back into the dataclasses fusion expects."""
    data = json.loads(Path(path).read_text())
    raw = data.get("observations", [])
    # Written alongside the JSON by the pipeline. Without it re-fusion can only
    # copy one frame's box; with it, it refits the same way the pipeline does.
    points = load_observation_points(path)

    observations: list[Observation] = []
    boxes: list[tuple[int, int, int, int] | None] = []

    for i, item in enumerate(raw):
        box = item.get("box_px")
        box = tuple(int(v) for v in box) if box else None
        boxes.append(box)
        observations.append(
            Observation(
                frame_idx=int(item["frame_idx"]),
                label=str(item["label"]),
                vlm_confidence=float(item.get("vlm_confidence", 0.5)),
                centroid=np.asarray(item["centroid"], dtype=float),
                obb=_orientedbox_from_dict(item["obb"]),
                n_points=int(item.get("n_points", 0)),
                support=float(item.get("support", 0.0)),
                box_px=box,
                points=points.get(int(item.get("id", i))),
            )
        )

    hw = data.get("image_hw") or [0, 0]
    return ObservationSet(
        room=data.get("room", ""),
        image_hw=(int(hw[0]), int(hw[1])),
        frames_labeled=[int(i) for i in data.get("frames_labeled", [])],
        observations=observations,
        boxes=boxes,
    )


def refuse(
    obs_set: ObservationSet,
    *,
    radius_floor: float = 0.30,
    radius_scale: float = 0.50,
    min_obb_iou: float = 0.10,
    min_confidence: float = 0.0,
    min_observations: int = 1,
    up: np.ndarray | None = None,
    floor_height: float | None = None,
) -> list[ObjectRecord]:
    """Re-cluster, then drop objects that fail the post-filters.

    Filtering happens *after* clustering, not before: an observation that is
    weak on its own may still be the third sighting that makes a cluster
    credible, so removing it up front would change the clustering rather than
    just the display.

    `up` and `floor_height` describe the room, not the thresholds being tuned;
    pass `scene_frame_of` so a re-fused box matches what the pipeline writes.
    """
    objects = cluster_observations(
        obs_set.observations,
        radius_floor=radius_floor,
        radius_scale=radius_scale,
        min_obb_iou=min_obb_iou,
        up=up,
        floor_height=floor_height,
    )
    return [
        o for o in objects
        if o.confidence >= min_confidence and o.n_observations >= min_observations
    ]


def scene_frame_of(room_dir: str | Path) -> tuple[np.ndarray | None, float | None]:
    """`(up, floor_height)` for a room on disk, from its `level.json`.

    Levelling already established which way is up and rewrote every artifact to
    match, so a levelled room's answer is exactly +Y with the floor at 0. Reading
    the record beats re-estimating: it is what the stored geometry was actually
    built against. An unlevelled room returns `(None, None)`, and re-fusion falls
    back to per-frame boxes rather than levelling against a guess.
    """
    record = load_level_record(room_dir)
    if not record or record.get("convention") != "y_up_floor_at_zero":
        return None, None
    return UP_VECTOR.copy(), 0.0


def radius_summary(
    obs_set: ObservationSet, radius_floor: float, radius_scale: float
) -> dict:
    """What the parameters actually amount to, in metres.

    Worth surfacing because the two knobs are not equally important: the merge
    radius is `max(floor, scale x mean OBB diagonal)`, so for furniture-sized
    objects the size term dominates and moving the floor from 0.3 to 1.0 does
    nothing at all. Showing the effective number turns a slider that "seems
    broken" into one whose behaviour is obvious.

    `merge_distance` is the stricter of the two gates: observations whose boxes
    do not overlap must be within half the radius.
    """
    diags = [o.obb.diagonal for o in obs_set.observations if o.obb.diagonal > 0]
    mean_diag = float(np.mean(diags)) if diags else 0.0
    radius = max(radius_floor, radius_scale * mean_diag)

    return {
        "mean_obb_diagonal": round(mean_diag, 3),
        "effective_radius": round(radius, 3),
        "merge_distance_no_overlap": round(0.5 * radius, 3),
        "driven_by": "size" if radius_scale * mean_diag > radius_floor else "floor",
    }


def objects_doc(
    room: str, objects: list[ObjectRecord], *, scale_verified: bool = False,
    n_frames_labeled: int = 0,
) -> dict:
    """The same shape the pipeline writes, so a saved re-fusion is indistinguishable."""
    return {
        "room": room,
        "units": "meters",
        "scale_verified": scale_verified,
        "n_frames_labeled": n_frames_labeled,
        "objects": [o.as_dict() for o in objects],
    }


def save_objects(path: str | Path, doc: dict) -> Path:
    """Write objects.json, keeping one generation of backup.

    Re-fusion is an experiment; overwriting the pipeline's output with no way
    back would make experimenting expensive.
    """
    path = Path(path)
    if path.exists():
        backup = path.with_suffix(".prev.json")
        backup.write_text(path.read_text())
    path.write_text(json.dumps(doc, indent=2))
    return path
