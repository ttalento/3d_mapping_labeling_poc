"""room3d command line.

    room3d extract     data/rooms/office/video.mp4 --room office
    room3d reconstruct out/office/frames --room office
    room3d level       --room office            # only for rooms built before levelling
    room3d label       out/office/frames.npz --room office [--direct]
    room3d refit       --room office            # redo the 3D boxes, no VLM calls
    room3d query       --room office "couch"     # name a thing, get its box
    room3d view        out/office
    room3d run         data/rooms/office/video.mp4 --room office

Stages are separate commands on purpose. Reconstruction takes tens of minutes on
CPU; relabeling should not require redoing it, and `frames.npz` is the interface
that makes the two halves independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def _out_dir(room: str) -> Path:
    return Path("out") / room


def cmd_extract(args) -> int:
    from .frames import extract_frames

    cfg = load_config(args.config).frames
    out = _out_dir(args.room) / "frames"
    frames = extract_frames(
        args.source, out,
        n_frames=args.n_frames or cfg.n_frames,
        blur_percentile=cfg.blur_percentile,
        dedup_correlation=cfg.dedup_correlation,
        min_frames=cfg.min_frames,
    )
    print(f"[extract] {len(frames)} frames -> {out}")
    for f in frames:
        print(f"          {f.path.name}  t={f.timestamp:7.2f}s  sharpness={f.sharpness:8.1f}")
    return 0


def cmd_reconstruct(args) -> int:
    from .reconstruct import reconstruct

    cfg = load_config(args.config).reconstruct
    frame_dir = Path(args.frames)
    paths = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(paths) < 2:
        print(f"error: need at least 2 frames in {frame_dir}", file=sys.stderr)
        return 1

    reconstruct(
        paths, _out_dir(args.room),
        checkpoint=args.checkpoint or cfg.checkpoint,
        image_size=cfg.image_size,
        scene_graph=args.scene_graph or cfg.scene_graph,
        niter=cfg.niter,
        lr=cfg.lr,
        schedule=cfg.schedule,
        min_conf_thr=cfg.min_conf_thr,
        device=cfg.device,
        scale_ref=args.scale_ref,
        allow_mixed=args.allow_mixed,
        level=not args.no_level,
    )
    return 0


def cmd_level(args) -> int:
    from .level import level_room

    out = _out_dir(args.room)
    if not (out / "frames.npz").exists():
        print(f"error: {out / 'frames.npz'} not found; reconstruct this room first",
              file=sys.stderr)
        return 1

    result = level_room(out, up=args.up)
    if result.estimate.confidence < 0.4 and not args.up:
        print("\nThat estimate is weak. Check the Floor plan tab, and if it is wrong "
              "re-run with an explicit axis, e.g.  room3d level --room "
              f"{args.room} --up -y", file=sys.stderr)
    return 0


def cmd_label(args) -> int:
    from .artifacts import load_frames_npz
    from .crew import run_labeling

    cfg = load_config(args.config).label
    if args.max_frames:
        cfg.max_frames = args.max_frames

    recon = load_frames_npz(args.npz)
    run_labeling(
        recon,
        _out_dir(args.room) / "objects.json",
        config=cfg,
        use_crew=not args.direct,
        room_name=args.room,
        scale_verified=args.scale_verified,
    )
    return 0


def cmd_refit(args) -> int:
    from .refit import refit_room

    out = _out_dir(args.room)
    cfg = load_config(args.config).label
    if args.min_observations is not None:
        cfg.min_observations = args.min_observations
    try:
        refit_room(out, config=cfg)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_query(args) -> int:
    from .query import commit_match, filter_by_certainty, query_room

    out = _out_dir(args.room)
    cfg = load_config(args.config).query
    overrides = {}
    if args.min_vote is not None:
        overrides["min_vote"] = args.min_vote

    detector = None
    if args.force or args.detector:
        detector = _build_detector(args)

    try:
        result = query_room(
            out, args.phrase,
            config=cfg, config_overrides=overrides or None,
            detector=detector, force=args.force,
            # cmd_query does its own printing below, numbered off the
            # certainty-filtered list. query_room's own numbering runs over
            # every match it found, before that filter is applied, and
            # --commit N must never index a different list than the one the
            # user just read.
            verbose=False,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    kept, dropped = result.matches, []
    if not args.all:
        kept, dropped = filter_by_certainty(
            result.matches, min_views=cfg.min_views, min_mean_vote=cfg.min_mean_vote
        )

    if args.json:
        print(json.dumps(_query_payload(result, kept, dropped), indent=2))
    else:
        _print_query_result(args.phrase, result, kept, dropped, cfg)

    if args.commit is not None:
        if args.commit < 1 or args.commit > len(kept):
            print(f"error: --commit is 1-indexed; this query returned "
                  f"{len(kept)} match(es)", file=sys.stderr)
            return 1
        try:
            committed = commit_match(
                out, kept[args.commit - 1],
                config=cfg, force=args.force_commit, verbose=not args.json,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        # A VLM-sourced match's Views carry no observation_id/object_id (see
        # `_vlm_views`), so `absorbed_object_ids` is always empty and this
        # commit can only ever add a new object -- never replace the
        # duplicates the flagship `--force --commit` flow exists to fix. Zero
        # removed reads the same either way in the ordinary "committed X;
        # removed 0" line above, so say the reason explicitly rather than
        # leave a user to infer it.
        if not args.json and result.source == "vlm" and not committed["removed"]:
            print(f"[query] note: {committed['object_id']} came from the VLM, "
                  f"not the cache, so it had no observation ids to absorb -- "
                  f"it was added, not used to replace an existing duplicate")

    return 0


def _print_query_result(phrase, result, kept, dropped, cfg) -> None:
    """The human-readable report, numbered `1..len(kept)` -- the same numbers
    `--commit N` indexes into, and the same list `_query_payload` reports as
    `matches`. Printing from here rather than from `query_room` is what keeps
    those three in lockstep instead of merely usually agreeing.
    """
    import numpy as np

    print(f"[query] {phrase!r} -> {len(kept)} match(es) from {result.source}")
    for i, m in enumerate(kept, 1):
        size = "unsupported" if m.obb is None else np.round(m.obb.extent, 3).tolist()
        print(f"[query]   {i}. {m.label:<16} score={m.score:.2f} "
              f"views={len(m.views)} extent={size}")

    if dropped:
        print(f"[query] {len(dropped)} match(es) hidden as too uncertain to place "
              f"(need {cfg.min_views}+ views agreeing at {cfg.min_mean_vote:.2f}); "
              f"--all shows them")
    for note in result.notes:
        print(f"[query] note: {note}")


def _query_payload(result, kept, dropped) -> dict:
    """The `--json` counterpart of `_print_query_result`.

    `matches[i]` is exactly what `--commit i+1` would write -- the same
    property the printed output has -- rather than the unfiltered result a
    machine caller could not safely correlate with `--commit`. Dropped
    matches are not reduced to a count: they are listed in full under
    `hidden`, because a caller must be able to see what got hidden, not just
    trust that something did.
    """
    return {
        "phrase": result.phrase,
        "source": result.source,
        "matches": [m.as_dict() for m in kept],
        "hidden": [m.as_dict() for m in dropped],
        "notes": list(result.notes),
    }


def _build_detector(args):
    """A detector only when one is actually needed -- building it loads .env."""
    from .vlm import GeminiDetector, load_env, resolve_model

    load_env()
    cfg = load_config(args.config).label
    return GeminiDetector(
        model=resolve_model(cfg.model),
        request_masks=False,
        max_retries=cfg.max_retries,
        min_interval_s=60.0 / max(cfg.max_rpm, 1),
    )


def cmd_view(args) -> int:
    from .viewer import view

    d = Path(args.dir)
    view(
        d / "scene.ply",
        d / "objects.json",
        d / "trajectory.txt",
        min_confidence=args.min_confidence,
    )
    return 0


def cmd_ui(args) -> int:
    from .webapp.server import serve

    vendor = Path(__file__).parent / "webapp" / "static" / "vendor" / "three.module.js"
    if not vendor.exists():
        print("error: frontend assets missing. Run:\n"
              "    uv run python scripts/fetch_vendor.py", file=sys.stderr)
        return 1

    print(f"room3d viewer -> http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port, out_dir=args.out, open_browser=args.open)
    return 0


def cmd_run(args) -> int:
    """Everything, end to end."""
    from .artifacts import load_frames_npz
    from .crew import run_labeling
    from .frames import extract_frames
    from .reconstruct import reconstruct

    cfg = load_config(args.config)
    out = _out_dir(args.room)

    frames = extract_frames(
        args.source, out / "frames",
        n_frames=args.n_frames or cfg.frames.n_frames,
        blur_percentile=cfg.frames.blur_percentile,
        dedup_correlation=cfg.frames.dedup_correlation,
        min_frames=cfg.frames.min_frames,
    )
    print(f"[run] extracted {len(frames)} frames")

    reconstruct(
        [f.path for f in frames], out,
        checkpoint=args.checkpoint or cfg.reconstruct.checkpoint,
        image_size=cfg.reconstruct.image_size,
        scene_graph=args.scene_graph or cfg.reconstruct.scene_graph,
        niter=cfg.reconstruct.niter,
        min_conf_thr=cfg.reconstruct.min_conf_thr,
        device=cfg.reconstruct.device,
        scale_ref=args.scale_ref,
        allow_mixed=args.allow_mixed,
        level=not args.no_level,
    )

    if args.no_label:
        print("[run] --no-label given; stopping after reconstruction")
        return 0

    run_labeling(
        load_frames_npz(out / "frames.npz"),
        out / "objects.json",
        config=cfg.label,
        use_crew=not args.direct,
        room_name=args.room,
        scale_verified=args.scale_ref is not None,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="room3d", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to a config YAML")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="video/folder -> sharp, deduplicated frames")
    e.add_argument("source")
    e.add_argument("--room", required=True)
    e.add_argument("--n-frames", type=int, default=None)
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("reconstruct", help="frames -> scene.ply, trajectory.txt, frames.npz")
    r.add_argument("frames")
    r.add_argument("--room", required=True)
    r.add_argument("--checkpoint", default=None)
    r.add_argument("--scene-graph", default=None, help='e.g. "swin-3" or "complete"')
    r.add_argument("--scale-ref", type=float, default=None,
                   help="rescale so the largest scene extent equals this many metres")
    r.add_argument("--allow-mixed", action="store_true",
                   help="if images have mixed orientations, keep only the majority")
    r.add_argument("--no-level", action="store_true",
                   help="keep the aligner's raw camera-anchored frame (leaves the "
                        "scene upside down; see `room3d level`)")
    r.set_defaults(func=cmd_reconstruct)

    g = sub.add_parser("level", help="stand an existing room upright: +Y up, floor at y=0")
    g.add_argument("--room", required=True)
    g.add_argument("--up", default=None,
                   help="skip estimation and declare the up axis: x, y, z, -x, -y, -z")
    g.set_defaults(func=cmd_level)

    l = sub.add_parser("label", help="frames.npz -> objects.json")
    l.add_argument("npz")
    l.add_argument("--room", required=True)
    l.add_argument("--direct", action="store_true",
                   help="call the tools in order without the agent loop")
    l.add_argument("--max-frames", type=int, default=None)
    l.add_argument("--scale-verified", action="store_true",
                   help="record that the metric scale was checked against a real measurement")
    l.set_defaults(func=cmd_label)

    f = sub.add_parser(
        "refit",
        help="rebuild 3D boxes from the detections already on disk (no VLM calls)",
    )
    f.add_argument("--room", required=True)
    f.add_argument("--min-observations", type=int, default=None,
                   help="drop objects seen fewer times than this (default 1, lossless)")
    f.set_defaults(func=cmd_refit)

    q = sub.add_parser("query", help="name an object, get its 3D box (no reconstruction)")
    q.add_argument("phrase", help='e.g. "couch" or "the couch by the window"')
    q.add_argument("--room", required=True)
    q.add_argument("--force", action="store_true",
                   help="ignore the cached detections and ask the VLM -- a "
                        "VLM-sourced match carries no observation/object ids, "
                        "so --commit can never make it absorb an existing "
                        "duplicate; it always adds a new object")
    q.add_argument("--detector", action="store_true",
                   help="allow VLM calls when the cache cannot answer")
    q.add_argument("--commit", type=int, default=None, metavar="N",
                   help="promote match N (1-indexed) into objects.json")
    q.add_argument("--min-vote", type=float, default=None,
                   help="fraction of the views that could see a point which "
                        "must agree it is inside the box (default 0.5). With "
                        "leave-one-out a point is judged by at most n_views-1 "
                        "frames, so this is a step function, not a smooth "
                        "knob -- see README.md's \"Querying one object\" for "
                        "the table. For a 2-view object no value changes "
                        "anything: one judge means every positive threshold "
                        "is unanimity")
    q.add_argument("--all", action="store_true",
                   help="include matches too uncertain to place")
    q.add_argument("--force-commit", action="store_true",
                   help="commit even if the match is below the certainty gate")
    q.add_argument("--json", action="store_true", help="machine-readable output")
    q.set_defaults(func=cmd_query)

    v = sub.add_parser("view", help="show the cloud with labeled boxes")
    v.add_argument("dir")
    v.add_argument("--min-confidence", type=float, default=0.0)
    v.set_defaults(func=cmd_view)

    u = sub.add_parser("ui", help="browser viewer: frames, cloud, floor plan, trajectory")
    u.add_argument("--port", type=int, default=8000)
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--out", default="out", help="directory holding the room outputs")
    u.add_argument("--open", action="store_true", help="open a browser window")
    u.set_defaults(func=cmd_ui)

    a = sub.add_parser("run", help="extract + reconstruct + label")
    a.add_argument("source")
    a.add_argument("--room", required=True)
    a.add_argument("--n-frames", type=int, default=None)
    a.add_argument("--checkpoint", default=None)
    a.add_argument("--scene-graph", default=None)
    a.add_argument("--scale-ref", type=float, default=None)
    a.add_argument("--allow-mixed", action="store_true")
    a.add_argument("--no-level", action="store_true")
    a.add_argument("--direct", action="store_true")
    a.add_argument("--no-label", action="store_true")
    a.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
