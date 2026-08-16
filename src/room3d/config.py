"""Typed config loaded from configs/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FramesConfig:
    n_frames: int = 24
    blur_percentile: float = 25.0        # reject frames below this Laplacian-variance pctile
    dedup_correlation: float = 0.985     # drop frame if corr with previous kept exceeds this
    min_frames: int = 4


@dataclass
class ReconstructConfig:
    checkpoint: str = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    image_size: int = 512
    scene_graph: str = "swin-3"          # O(N) pairs; "complete" is O(N^2) and CPU-hostile
    niter: int = 300
    lr: float = 0.01
    schedule: str = "cosine"
    min_conf_thr: float = 3.0
    device: str = "cpu"


@dataclass
class LabelConfig:
    model: str = "gemini-3.6-flash"   # MODEL/GEMINI_MODEL in .env overrides this
    max_frames: int = 24
    # Off by default: current Gemini models either omit the mask or blow the
    # output token budget returning it. Boxes + depth clustering is the reliable
    # path. See the module docstring in vlm.py for the measurements.
    use_masks: bool = False
    mask_erode_px: int = 3
    min_mask_points: int = 40            # below this, drop the detection
    depth_cluster_eps: float = 0.15      # metres, 1-D DBSCAN along the ray
    merge_radius_floor: float = 0.30     # metres
    merge_radius_scale: float = 0.50     # fraction of mean OBB diagonal
    min_obb_iou: float = 0.10
    max_retries: int = 4
    # Gemini's free tier allows ~15 requests/minute per model. The crew path
    # spends calls on agent reasoning *as well as* detection, so both have to
    # share one budget or the run dies partway through with a 429.
    max_rpm: int = 10


@dataclass
class Config:
    frames: FramesConfig = field(default_factory=FramesConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    label: LabelConfig = field(default_factory=LabelConfig)


def _build(cls, data: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        frames=_build(FramesConfig, raw.get("frames", {})),
        reconstruct=_build(ReconstructConfig, raw.get("reconstruct", {})),
        label=_build(LabelConfig, raw.get("label", {})),
    )
