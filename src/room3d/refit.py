"""Rebuild a labelled room's 3D geometry from detections already on disk.

Detection is the only expensive step in labeling: one VLM call per frame, against
a quota. Everything after it -- lifting pixels to 3D, fitting boxes, merging
observations into objects -- is arithmetic over arrays that are already stored.
So an improvement to the geometry should never require paying for detection
again, and `observations.json` records exactly what is needed to avoid it: the
pixel box, the label and the confidence the VLM returned for every detection.

This re-runs projection and fusion over those, writes the results back, and never
constructs a detector. It is also the honest way to compare two fitting
strategies, since both see byte-identical detections.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import load_frames_npz, save_observation_points
from .config import LabelConfig
from .fusion import cluster_observations
from .level import estimate_up
from .projection import box_to_mask, camera_center_from_pose, project_detection


def refit_room(
    room_dir: str | Path,
    *,
    config: LabelConfig | None = None,
    verbose: bool = True,
) -> dict:
    """Re-project and re-fuse a room from its stored detections. Returns a summary."""
    room_dir = Path(room_dir)
    npz = room_dir / "frames.npz"
    obs_path = room_dir / "observations.json"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found; reconstruct this room first")
    if not obs_path.exists():
        raise FileNotFoundError(f"{obs_path} not found; label this room first")

    config = config or LabelConfig()
    recon = load_frames_npz(npz)
    doc = json.loads(obs_path.read_text())
    stored = doc.get("observations", [])

    up, floor_height = _scene_frame(recon, config, verbose)
    h, w = recon.image_hw

    observations = []
    skipped = 0
    for item in stored:
        box = item.get("box_px")
        frame_idx = int(item["frame_idx"])
        if not box or frame_idx >= recon.n_frames:
            # No pixel box means nothing to re-project. The old 3D record cannot
            # be improved and must not be silently carried forward as if it had.
            skipped += 1
            continue

        box_px = tuple(int(v) for v in box)
        obs = project_detection(
            box_to_mask(box_px, h, w),
            recon.pts3d[frame_idx],
            recon.conf_mask[frame_idx],
            camera_center_from_pose(recon.poses[frame_idx]),
            frame_idx=frame_idx,
            label=str(item["label"]),
            vlm_confidence=float(item.get("vlm_confidence", 0.5)),
            erode_px=config.mask_erode_px,
            depth_eps=config.depth_cluster_eps,
            min_points=config.min_mask_points,
            box_px=box_px,
            up=up,
            max_points=config.max_points_per_observation,
            obb_percentile=config.obb_percentile,
        )
        if obs is None:
            skipped += 1
        else:
            observations.append(obs)

    objects = cluster_observations(
        observations,
        radius_floor=config.merge_radius_floor,
        radius_scale=config.merge_radius_scale,
        min_obb_iou=config.min_obb_iou,
        up=up,
        floor_height=floor_height if config.floor_snap_threshold > 0 else None,
        floor_snap_threshold=config.floor_snap_threshold,
        obb_percentile=config.obb_percentile,
        max_pooled_points=config.max_pooled_points,
    )

    previous = json.loads((room_dir / "objects.json").read_text()) if (
        room_dir / "objects.json"
    ).exists() else {}
    room_name = previous.get("room") or doc.get("room") or room_dir.name

    _write(room_dir, obs_path, doc, room_name, previous, observations, objects)

    summary = {
        "room": room_name,
        "n_observations": len(observations),
        "n_skipped": skipped,
        "n_objects": len(objects),
        "previous_n_objects": len(previous.get("objects", [])),
        "levelled": up is not None,
    }
    if verbose:
        print(f"[refit] {len(observations)} observations ({skipped} skipped) "
              f"-> {len(objects)} objects "
              f"(was {summary['previous_n_objects']})")
    return summary


def _scene_frame(recon, config: LabelConfig, verbose: bool):
    """`(up, floor_height)`, or `(None, None)` when levelling cannot be trusted."""
    if not config.gravity_aligned_boxes:
        return None, None

    estimate = estimate_up(recon.pts3d[recon.conf_mask], recon.poses)
    if estimate.confidence < config.min_level_confidence:
        if verbose:
            print(f"[refit] up estimate too weak ({estimate.confidence:.2f}); "
                  "keeping PCA boxes")
        return None, None

    if verbose:
        print(f"[refit] up={np.round(estimate.up, 3).tolist()} "
              f"({estimate.source}, confidence {estimate.confidence:.2f})")
    return estimate.up, float(estimate.floor_offset)


def _write(room_dir, obs_path, doc, room_name, previous, observations, objects) -> None:
    """Rewrite objects.json and observations.json, keeping one generation back.

    Both get a `.prev.json` copy first. `observations.json` carries the pixel
    boxes and labels the VLM returned, which are the only part of a run that
    cannot be recomputed -- rewriting it in place would make a bad refit
    unrecoverable without spending the quota again.

    The previous run's metadata is carried over rather than re-derived: whether
    the metric scale was verified against a real measurement is a fact about the
    capture, and nothing here re-measures it.
    """
    owner = {oid: obj.id for obj in objects for oid in obj.observation_ids}

    for path in (room_dir / "objects.json", obs_path):
        if path.exists():
            path.with_suffix(".prev.json").write_text(path.read_text())

    objects_path = room_dir / "objects.json"
    objects_path.write_text(json.dumps({
        "room": room_name,
        "units": "meters",
        "scale_verified": previous.get("scale_verified", False),
        "n_frames_labeled": previous.get(
            "n_frames_labeled", len({o.frame_idx for o in observations})
        ),
        "objects": [o.as_dict() for o in objects],
    }, indent=2))

    obs_path.write_text(json.dumps({
        "room": room_name,
        "image_hw": doc.get("image_hw"),
        "frames_labeled": doc.get("frames_labeled", []),
        "observations": [
            {"id": i, "object_id": owner.get(i), **obs.as_dict()}
            for i, obs in enumerate(observations)
        ],
    }, indent=2))
    save_observation_points(obs_path, observations)
