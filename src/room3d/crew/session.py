"""Shared state the CrewAI tools operate on.

Agents exchange short summaries; the arrays stay here. Passing pointmaps through
an LLM's context would be both ruinous and pointless -- the model's job is to
decide *which* frames to look at and *what things are called*, not to carry
megabytes of geometry between steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..artifacts import Reconstruction
from ..config import LabelConfig
from ..fusion import ObjectRecord
from ..projection import Observation
from ..vlm import Detection


@dataclass
class LabelingSession:
    recon: Reconstruction
    config: LabelConfig

    selected_frames: list[int] = field(default_factory=list)
    detections: dict[int, list[Detection]] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    objects: list[ObjectRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.notes.append(message)
        print(f"  {message}")

    @property
    def observed_labels(self) -> list[str]:
        return [d.label for dets in self.detections.values() for d in dets]

    def summary(self) -> dict:
        return {
            "frames_available": self.recon.n_frames,
            "frames_selected": len(self.selected_frames),
            "detections": sum(len(d) for d in self.detections.values()),
            "observations": len(self.observations),
            "objects": len(self.objects),
        }


def select_covering_frames(recon: Reconstruction, max_frames: int) -> list[int]:
    """Pick frames that cover the room, not merely the timeline.

    Farthest-point sampling over camera centres. A walkthrough often lingers in
    one spot and sweeps past another; uniform temporal sampling inherits that
    bias, and spending VLM calls on ten views of the same corner buys nothing.
    Seeded from the frame nearest the centroid so the first pick is central
    rather than an arbitrary endpoint.
    """
    n = recon.n_frames
    if max_frames >= n:
        return list(range(n))

    centres = recon.poses[:, :3, 3].astype(np.float64)
    if not np.isfinite(centres).all():
        return list(range(0, n, max(1, n // max_frames)))[:max_frames]

    seed = int(np.argmin(np.linalg.norm(centres - centres.mean(axis=0), axis=1)))
    chosen = [seed]
    distances = np.linalg.norm(centres - centres[seed], axis=1)

    while len(chosen) < max_frames:
        distances[chosen] = -1.0            # never re-pick
        nxt = int(np.argmax(distances))
        if distances[nxt] < 0:
            break
        chosen.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(centres - centres[nxt], axis=1))

    # Farthest-point sampling stalls when cameras are coincident -- which is
    # exactly what a person standing still produces. Top up from the unused
    # frames, spread evenly, so a static stretch cannot starve the budget.
    if len(chosen) < max_frames:
        spare = [i for i in range(n) if i not in set(chosen)]
        need = max_frames - len(chosen)
        if spare and need > 0:
            step = max(1, len(spare) // need)
            chosen.extend(spare[::step][:need])

    return sorted(set(chosen))
