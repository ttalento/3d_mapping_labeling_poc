"""Stage 2: MASt3R weights driven by DUSt3R's global aligner, on CPU.

We deliberately do not run MASt3R-SLAM. It compiles custom CUDA kernels at
install time and has no CPU path, and this POC is offline batch work that needs
neither live tracking nor loop closure.

What we do NOT drop is pose estimation. Global alignment *is* the pose solver:
`get_im_poses()` and `get_focals()` fall out of the same optimisation that
produces `get_pts3d()`. Reconstruction and camera recovery are one problem.

The output that matters most downstream is `frames.npz` -- a per-frame,
world-frame pointmap plus its pose. That is what turns the labeling stage's
2D->3D projection into an array lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: F401  (used in _enforce_uniform_shape messages)

import numpy as np

from . import vendor
from .artifacts import Reconstruction, save_frames_npz, save_ply, save_trajectory_tum


@dataclass
class ReconstructResult:
    recon: Reconstruction
    ply_path: Path
    traj_path: Path
    npz_path: Path
    n_points: int
    scene_extent: np.ndarray
    up_estimate: dict | None = None


def _load_model(checkpoint: str | Path, device: str):
    vendor.ensure()
    import mast3r.utils.path_to_dust3r  # noqa: F401  (registers dust3r on sys.path)
    from mast3r.model import AsymmetricMASt3R

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}\n"
            "Download MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth into checkpoints/"
        )
    return AsymmetricMASt3R.from_pretrained(str(checkpoint)).to(device).eval()


def _intrinsics_from_scene(scene, n: int) -> np.ndarray:
    """(N, 3, 3) pinhole matrices for the reconstruction-resolution images."""
    try:
        K = scene.get_intrinsics().detach().cpu().numpy()
        if K.shape == (n, 3, 3):
            return K.astype(np.float32)
    except (AttributeError, NotImplementedError):
        pass

    focals = scene.get_focals().detach().cpu().numpy().reshape(n, -1)
    pps = scene.get_principal_points().detach().cpu().numpy().reshape(n, 2)

    K = np.zeros((n, 3, 3), dtype=np.float32)
    K[:, 0, 0] = focals[:, 0]
    K[:, 1, 1] = focals[:, -1]        # square pixels when only one focal is predicted
    K[:, 0, 2] = pps[:, 0]
    K[:, 1, 2] = pps[:, 1]
    K[:, 2, 2] = 1.0
    return K


def reconstruct(
    image_paths: list[str | Path],
    out_dir: str | Path,
    *,
    checkpoint: str | Path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    image_size: int = 512,
    scene_graph: str = "swin-3",
    niter: int = 300,
    lr: float = 0.01,
    schedule: str = "cosine",
    min_conf_thr: float = 3.0,
    device: str = "cpu",
    scale_ref: float | None = None,
    allow_mixed: bool = False,
    level: bool = True,
    verbose: bool = True,
) -> ReconstructResult:
    """Reconstruct a scene and write scene.ply, trajectory.txt and frames.npz.

    `scene_graph` defaults to a sliding window. The alternative, "complete", is
    O(N^2) pairs -- 24 frames means 552 forward passes, which is hours on CPU.
    A walkthrough is temporally ordered, so a window is also the correct graph,
    not merely the affordable one.

    `scale_ref`, if given, rescales the scene so its largest extent equals this
    many metres. The metric checkpoint is usually close already; this is the
    escape hatch for when global alignment's per-pair scale factors have drifted.

    `level` stands the result upright. The aligner's world frame is a camera
    frame, whose +Y points *down* the image, so the raw output is upside down in
    any +Y-up viewer. Levelling is a rigid transform -- it changes no geometry,
    only which direction the numbers call up. See `level.py`.
    """
    vendor.ensure()
    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [str(p) for p in image_paths]
    if len(paths) < 2:
        raise ValueError(f"need at least 2 images, got {len(paths)}")

    model = _load_model(checkpoint, device)
    images = load_images(paths, size=image_size, verbose=verbose)

    # Validate shape uniformity BEFORE inference, not after. Mixed portrait and
    # landscape input yields different (H, W) per image, which cannot be stacked
    # into the (N, H, W, ...) arrays the rest of the pipeline is built on.
    # Discovering that after 30 minutes of forward passes is unacceptable.
    images, source_indices = _enforce_uniform_shape(
        images, paths, allow_mixed=allow_mixed, verbose=verbose
    )

    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if verbose:
        print(f"[reconstruct] {len(images)} images -> {len(pairs)} pairs ({scene_graph})")

    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=verbose)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    scene.min_conf_thr = min_conf_thr

    n = len(images)
    imgs = np.stack([(np.asarray(im) * 255).astype(np.uint8) for im in scene.imgs])
    pts3d = np.stack([p.detach().cpu().numpy() for p in scene.get_pts3d()]).astype(np.float32)
    conf_mask = np.stack([m.detach().cpu().numpy() for m in scene.get_masks()]).astype(bool)
    poses = scene.get_im_poses().detach().cpu().numpy().astype(np.float32)
    intrinsics = _intrinsics_from_scene(scene, n)

    if scale_ref is not None:
        pts3d, poses, factor = _rescale(pts3d, poses, scale_ref)
        if verbose:
            print(f"[reconstruct] applied scale_ref: x{factor:.4f}")

    recon = Reconstruction(
        images=imgs,
        pts3d=pts3d,
        conf_mask=conf_mask,
        poses=poses,
        intrinsics=intrinsics,
        # Indices into the *original* frame set, so a labeled object's seen_in
        # can still be traced back to a file on disk after filtering.
        frame_ids=np.asarray(source_indices, dtype=np.int32),
    )

    up_estimate = None
    if level:
        from .level import level_reconstruction

        recon, transform, estimate = level_reconstruction(recon)
        pts3d, poses = recon.pts3d, recon.poses
        up_estimate = estimate.as_dict()
        angle = float(
            np.degrees(np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1)))
        )
        if verbose:
            print(f"[reconstruct] levelled: up was {np.round(estimate.up, 3).tolist()} "
                  f"({estimate.source}, confidence {estimate.confidence:.2f}); "
                  f"rotated {angle:.1f} deg, floor -> y=0")
            for note in estimate.notes:
                print(f"[reconstruct] note: {note}")

    ply_path = out_dir / "scene.ply"
    traj_path = out_dir / "trajectory.txt"
    npz_path = out_dir / "frames.npz"

    points = pts3d[conf_mask]
    colors = imgs[conf_mask]
    save_ply(ply_path, points, colors)
    save_trajectory_tum(traj_path, poses)
    save_frames_npz(npz_path, recon)

    if level:
        import json

        (out_dir / "level.json").write_text(
            json.dumps(
                {
                    "convention": "y_up_floor_at_zero",
                    "applied": transform.tolist(),
                    "cumulative": transform.tolist(),
                    "rotation_deg": round(angle, 2),
                    "estimate": up_estimate,
                },
                indent=2,
            )
        )

    extent = points.max(axis=0) - points.min(axis=0) if len(points) else np.zeros(3)
    if verbose:
        kept, total = int(conf_mask.sum()), int(conf_mask.size)
        print(f"[reconstruct] {kept}/{total} confident points ({100 * kept / total:.1f}%)")
        print(f"[reconstruct] scene extent (m): {np.round(extent, 2).tolist()}")
        print(f"[reconstruct] wrote {ply_path.name}, {traj_path.name}, {npz_path.name}")

    return ReconstructResult(
        recon=recon,
        ply_path=ply_path,
        traj_path=traj_path,
        npz_path=npz_path,
        n_points=len(points),
        scene_extent=extent,
        up_estimate=up_estimate,
    )


def _enforce_uniform_shape(
    images: list, paths: list[str] | None = None, *, allow_mixed: bool, verbose: bool
) -> tuple[list, list[int]]:
    """Keep the pipeline's (N, H, W, ...) contract intact.

    Returns the surviving images and their indices in the original input, so
    provenance back to the extracted frame files is not lost.

    Frames from one video are always uniform, so this only bites on folders of
    mixed-orientation photos. Rather than make every downstream array ragged,
    we refuse -- unless explicitly told to keep the majority orientation.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for i, im in enumerate(images):
        h, w = tuple(im["img"].shape[-2:])
        groups.setdefault((h, w), []).append(i)

    if len(groups) == 1:
        return images, list(range(len(images)))

    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    summary = ", ".join(
        f"{h}x{w}: {len(idx)} image{'s' if len(idx) != 1 else ''}" for (h, w), idx in ordered
    )

    if not allow_mixed:
        # Name the odd ones out; "one of your photos is sideways" is only
        # actionable if you know which.
        minority = [i for _, idx in ordered[1:] for i in idx]
        names = (
            "\nOdd ones out: "
            + ", ".join(Path(paths[i]).name for i in sorted(minority)[:8])
            if paths
            else ""
        )
        raise ValueError(
            f"Input images have mixed shapes after resizing ({summary}).\n"
            "This usually means the folder mixes portrait and landscape photos. "
            "Frames extracted from a single video are always uniform."
            f"{names}\n"
            "Either supply images of one orientation, or pass --allow-mixed to "
            "keep only the majority orientation."
        )

    keep = sorted(max(groups.values(), key=len))
    if len(keep) < 2:
        raise ValueError(
            f"Only {len(keep)} image remains after filtering to one orientation "
            f"({summary}); need at least 2."
        )

    if verbose:
        h, w = tuple(images[keep[0]]["img"].shape[-2:])
        print(f"[reconstruct] mixed shapes ({summary}); keeping {len(keep)} images "
              f"at {h}x{w}, dropping {len(images) - len(keep)}")

    # load_images stamps each entry with idx/instance, and the global aligner
    # asserts the edge indices form a contiguous range. Dropping images leaves
    # gaps (0,1,2,4,5), so the survivors must be renumbered.
    kept = []
    for new_idx, old_idx in enumerate(keep):
        image = dict(images[old_idx])
        image["idx"] = new_idx
        image["instance"] = str(new_idx)
        kept.append(image)
    return kept, keep


def _rescale(
    pts3d: np.ndarray, poses: np.ndarray, target_largest_extent: float
) -> tuple[np.ndarray, np.ndarray, float]:
    finite = np.isfinite(pts3d).all(axis=-1)
    pts = pts3d[finite]
    if not len(pts):
        return pts3d, poses, 1.0

    current = float((pts.max(axis=0) - pts.min(axis=0)).max())
    if current <= 1e-9:
        return pts3d, poses, 1.0

    factor = target_largest_extent / current
    scaled_poses = poses.copy()
    scaled_poses[:, :3, 3] *= factor
    return pts3d * factor, scaled_poses, factor
