"""Gate 0: is MASt3R + DUSt3R global alignment actually viable on this CPU?

Everything in the plan's schedule hangs off the per-pair forward time. This
measures it instead of assuming it. Run before trusting any time estimate.

    uv run python scripts/gate0_spike.py --n 6 --image-size 512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room3d import vendor  # noqa: E402

vendor.ensure()

import mast3r.utils.path_to_dust3r  # noqa: E402,F401  (registers dust3r on the path)
import torch  # noqa: E402
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner  # noqa: E402
from dust3r.image_pairs import make_pairs  # noqa: E402
from dust3r.inference import inference  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402
from mast3r.model import AsymmetricMASt3R  # noqa: E402

DEFAULT_CKPT = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DEFAULT_IMAGES = "third_party/mast3r/assets/NLE_tower"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--scene-graph", default="swin-3")
    ap.add_argument("--niter", type=int, default=300)
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    print(f"torch {torch.__version__} | threads={torch.get_num_threads()} | device=cpu")

    paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.n]
    if len(paths) < 2:
        print(f"need >=2 images in {args.images}, found {len(paths)}")
        return 1
    print(f"images: {len(paths)} from {args.images}")

    t0 = time.perf_counter()
    model = AsymmetricMASt3R.from_pretrained(args.ckpt).to("cpu").eval()
    t_load = time.perf_counter() - t0
    print(f"[1/4] model load        {t_load:7.1f} s")

    t0 = time.perf_counter()
    images = load_images([str(p) for p in paths], size=args.image_size, verbose=False)
    t_imgs = time.perf_counter() - t0
    print(f"[2/4] image load        {t_imgs:7.1f} s   shape={tuple(images[0]['img'].shape)}")

    pairs = make_pairs(images, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)
    print(f"      pairs({args.scene_graph}) = {len(pairs)}"
          f"   [complete would be {len(images) * (len(images) - 1)}]")

    t0 = time.perf_counter()
    with torch.no_grad():
        output = inference(pairs, model, "cpu", batch_size=1, verbose=False)
    t_inf = time.perf_counter() - t0
    per_pair = t_inf / max(len(pairs), 1)
    print(f"[3/4] inference         {t_inf:7.1f} s   ({per_pair:.1f} s/pair)")

    t0 = time.perf_counter()
    scene = global_aligner(output, device="cpu", mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=args.niter, schedule="cosine", lr=0.01)
    t_ga = time.perf_counter() - t0
    print(f"[4/4] global alignment  {t_ga:7.1f} s   ({args.niter} iters)")

    pts3d = scene.get_pts3d()
    poses = scene.get_im_poses()
    focals = scene.get_focals()
    masks = scene.get_masks()

    print("\n--- outputs ---")
    print(f"pts3d      {len(pts3d)} x {tuple(pts3d[0].shape)}  (world frame)")
    print(f"poses      {tuple(poses.shape)}")
    print(f"focals     {[round(float(f), 1) for f in focals.flatten()]}")
    kept = sum(int(m.sum()) for m in masks)
    total = sum(int(m.numel()) for m in masks)
    print(f"confident  {kept}/{total} points ({100 * kept / total:.1f}%)")

    centres = poses[:, :3, 3].detach().cpu().numpy()
    span = centres.max(axis=0) - centres.min(axis=0)
    print(f"trajectory span (m)  {[round(float(v), 2) for v in span]}")

    all_pts = torch.cat([p.reshape(-1, 3) for p in pts3d]).detach().cpu().numpy()
    extent = all_pts.max(axis=0) - all_pts.min(axis=0)
    print(f"scene extent (m)     {[round(float(v), 2) for v in extent]}")

    # --- the projection to 24 frames, which is what the plan actually needs ---
    print("\n--- extrapolation ---")
    window = int(args.scene_graph.split("-")[1]) if "-" in args.scene_graph else 3
    for n in (12, 24):
        est_pairs = 2 * sum(min(window, n - 1 - i) for i in range(n))
        est = (est_pairs * per_pair + t_ga * (n / len(images))) / 60.0
        print(f"  {n:2d} frames -> ~{est_pairs:3d} pairs -> ~{est:5.1f} min")

    print("\nGATE 0 VERDICT: "
          + ("PASS" if per_pair <= 30 else "FAIL — re-plan, see PLAN.md")
          + f" (threshold 30 s/pair, measured {per_pair:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
