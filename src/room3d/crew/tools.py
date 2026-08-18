"""CrewAI tools over the pipeline's deterministic functions.

Every tool returns a short text summary. The heavy arrays never leave the
session. Note what is and is not delegated to the model: the agent chooses which
frames to spend calls on and what things are called; it never computes a
coordinate. Geometry that an LLM could "reason" its way through wrongly stays in
tested Python.
"""

from __future__ import annotations

import json

import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..fusion import cluster_observations
from ..projection import (
    box_to_mask,
    camera_center_from_pose,
    decode_gemini_mask,
    descale_box,
    project_detection,
)
from ..vlm import GeminiDetector, VLMError
from .semantics import build_synonym_resolver
from .session import LabelingSession, select_covering_frames


# Every tool declares at least one argument on purpose. Gemini's function-calling
# API returns an empty completion when handed a function declaration with an
# empty parameters object, which surfaces from CrewAI as the opaque
# "Invalid response from LLM call - None or empty" before any tool ever runs.


class SelectFramesArgs(BaseModel):
    max_frames: int = Field(default=0, description="Budget of frames to label. 0 = config default.")


class DetectObjectsArgs(BaseModel):
    frame_indices: list[int] = Field(
        default_factory=list,
        description="Frames to inspect. Empty means use the frames already selected.",
    )


class ProjectDetectionsArgs(BaseModel):
    min_points: int = Field(
        default=0,
        description=(
            "Minimum confident 3D points required to keep a detection. "
            "0 = use the configured default."
        ),
    )


class SelectFramesTool(BaseTool):
    name: str = "select_frames"
    description: str = (
        "Choose which reconstructed frames to spend VLM calls on, maximising spatial "
        "coverage of the room using the camera poses. Returns the chosen frame indices."
    )
    args_schema: type[BaseModel] = SelectFramesArgs

    def __init__(self, session: LabelingSession, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    def _run(self, max_frames: int = 0) -> str:
        s = self._session
        budget = max_frames or s.config.max_frames
        s.selected_frames = select_covering_frames(s.recon, budget)

        centres = s.recon.poses[s.selected_frames][:, :3, 3]
        span = (centres.max(axis=0) - centres.min(axis=0)) if len(centres) else np.zeros(3)
        s.log(
            f"[frames] selected {len(s.selected_frames)}/{s.recon.n_frames}: "
            f"{s.selected_frames}, camera span {np.round(span, 2).tolist()} m"
        )
        return f"Selected {len(s.selected_frames)} frames: {s.selected_frames}"


class DetectObjectsTool(BaseTool):
    name: str = "detect_objects"
    description: str = (
        "Run the vision model over every selected frame to find objects, returning "
        "labels with 2D boxes. Call select_frames first."
    )
    args_schema: type[BaseModel] = DetectObjectsArgs

    def __init__(self, session: LabelingSession, detector: GeminiDetector, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._detector = detector

    def _run(self, frame_indices: list[int] | None = None) -> str:
        s = self._session
        if frame_indices:
            s.selected_frames = [i for i in frame_indices if 0 <= i < s.recon.n_frames]
        if not s.selected_frames:
            s.selected_frames = select_covering_frames(s.recon, s.config.max_frames)

        failures = 0
        for i in s.selected_frames:
            try:
                detections = self._detector.detect(s.recon.images[i])
            except VLMError as exc:
                failures += 1
                s.log(f"[detect] frame {i} failed: {exc}")
                continue

            s.detections[i] = detections
            labels = ", ".join(sorted({d.label for d in detections})) or "(none)"
            s.log(f"[detect] frame {i:>3}: {len(detections):>2} objects — {labels}")

        total = sum(len(d) for d in s.detections.values())
        vocabulary = sorted(set(s.observed_labels))
        return (
            f"Detected {total} objects across {len(s.detections)} frames "
            f"({failures} frames failed). Vocabulary: {vocabulary}"
        )


class ProjectDetectionsTool(BaseTool):
    name: str = "project_detections"
    description: str = (
        "Lift every 2D detection into 3D using the per-frame world-frame pointmaps, "
        "producing a 3D centroid and oriented box per observation. Discards detections "
        "with too little confident geometry behind them."
    )
    args_schema: type[BaseModel] = ProjectDetectionsArgs

    def __init__(self, session: LabelingSession, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    def _run(self, min_points: int = 0) -> str:
        s = self._session
        cfg = s.config
        min_points = min_points or cfg.min_mask_points
        h, w = s.recon.image_hw
        up, _ = s.gravity

        s.observations = []
        dropped_geometry = 0
        used_mask = used_box = 0

        for frame_idx, detections in s.detections.items():
            pts3d = s.recon.pts3d[frame_idx]
            conf = s.recon.conf_mask[frame_idx]
            centre = camera_center_from_pose(s.recon.poses[frame_idx])

            for det in detections:
                box_px = descale_box(det.box_2d, h, w)

                mask = None
                if cfg.use_masks and det.mask_b64:
                    try:
                        mask = decode_gemini_mask(det.mask_b64, box_px, h, w)
                        used_mask += 1
                    except Exception:  # noqa: BLE001 - a bad mask falls back to its box
                        mask = None
                if mask is None or not mask.any():
                    mask = box_to_mask(box_px, h, w)
                    used_box += 1

                obs = project_detection(
                    mask, pts3d, conf, centre,
                    frame_idx=frame_idx,
                    label=det.label,
                    vlm_confidence=det.confidence,
                    erode_px=cfg.mask_erode_px,
                    depth_eps=cfg.depth_cluster_eps,
                    min_points=min_points,
                    box_px=box_px,
                    up=up,
                    max_points=cfg.max_points_per_observation,
                    obb_percentile=cfg.obb_percentile,
                )
                if obs is None:
                    dropped_geometry += 1
                else:
                    s.observations.append(obs)

        s.log(
            f"[project] {len(s.observations)} observations "
            f"({used_mask} from masks, {used_box} from boxes); "
            f"{dropped_geometry} dropped for insufficient confident geometry"
        )
        return (
            f"Projected {len(s.observations)} observations into 3D; "
            f"dropped {dropped_geometry} unsupported detections."
        )


class ClusterObservationsArgs(BaseModel):
    use_llm_synonyms: bool = Field(
        default=True,
        description="Ask the LLM to group synonymous labels before clustering.",
    )


class ClusterObservationsTool(BaseTool):
    name: str = "cluster_observations"
    description: str = (
        "Merge per-frame 3D observations into unique objects. Geometry proposes the "
        "clusters using a size-adaptive radius; the LLM only decides which labels are "
        "synonyms. Returns the final object list as JSON."
    )
    args_schema: type[BaseModel] = ClusterObservationsArgs

    def __init__(self, session: LabelingSession, llm=None, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._llm = llm

    def _run(self, use_llm_synonyms: bool = True) -> str:
        s = self._session
        # Always return JSON, including the empty case. A tool that returns an
        # array on success and an English sentence on failure forces every
        # caller to guess which it got.
        if not s.observations:
            s.objects = []
            s.log("[cluster] no observations to cluster")
            return "[]"

        call_llm = None
        if use_llm_synonyms and self._llm is not None:
            call_llm = lambda prompt: self._llm.call(prompt)  # noqa: E731

        resolver = build_synonym_resolver(s.observed_labels, call_llm)
        up, floor_height = s.gravity

        s.objects = cluster_observations(
            s.observations,
            label_compatible=resolver.compatible,
            canonicalize=resolver.canonicalize,
            radius_floor=s.config.merge_radius_floor,
            radius_scale=s.config.merge_radius_scale,
            min_obb_iou=s.config.min_obb_iou,
            up=up,
            floor_height=floor_height if s.config.floor_snap_threshold > 0 else None,
            floor_snap_threshold=s.config.floor_snap_threshold,
            obb_percentile=s.config.obb_percentile,
            max_pooled_points=s.config.max_pooled_points,
        )

        s.log(f"[cluster] {len(s.observations)} observations -> {len(s.objects)} objects")
        for obj in s.objects:
            s.log(
                f"[cluster]   {obj.label:<18} conf={obj.confidence:.2f} "
                f"seen={obj.n_observations}x at {np.round(obj.centroid, 2).tolist()}"
            )

        return json.dumps([o.as_dict() for o in s.objects], indent=2)


def build_tools(session: LabelingSession, detector: GeminiDetector, llm=None) -> list[BaseTool]:
    return [
        SelectFramesTool(session),
        DetectObjectsTool(session, detector),
        ProjectDetectionsTool(session),
        ClusterObservationsTool(session, llm),
    ]
