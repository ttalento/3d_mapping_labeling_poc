"""FastAPI app serving the viewer.

Read-only over out/, plus two mutating actions: starting a run, and saving a
re-fusion (which keeps a backup). Binds 127.0.0.1 and has no auth -- it is a
local tool and must not be exposed.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..level import load_level_record
from . import floorplan as fp
from . import rooms as R
from .refuse import (
    load_observations,
    objects_doc,
    radius_summary,
    refuse,
    save_objects,
    scene_frame_of,
)
from .runs import RunManager

STATIC = Path(__file__).parent / "static"

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


# These must live at module scope. `from __future__ import annotations` makes
# every annotation a string, and FastAPI resolves those against the module
# globals -- a model defined inside create_app() is invisible there, so FastAPI
# silently falls back to treating the parameter as a query field and every POST
# fails with "field required: query.req".
class RefuseRequest(BaseModel):
    radius_floor: float = Field(0.30, ge=0.01, le=5.0)
    radius_scale: float = Field(0.50, ge=0.0, le=5.0)
    min_obb_iou: float = Field(0.10, ge=0.0, le=1.0)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)
    min_observations: int = Field(1, ge=1, le=50)
    save: bool = False


class RunRequest(BaseModel):
    source: str
    room: str
    n_frames: int | None = Field(None, ge=2, le=200)
    direct: bool = True
    scene_graph: str | None = None
    allow_mixed: bool = False
    no_label: bool = False


def create_app(out_dir: str | Path = "out", project_root: str | Path = ".") -> FastAPI:
    out_dir = Path(out_dir)
    project_root = Path(project_root)

    app = FastAPI(title="room3d viewer", docs_url="/api/docs")
    app.state.out_dir = out_dir
    app.state.project_root = project_root
    app.state.runs = RunManager(cwd=project_root)
    # PLY reads are ~100 ms on a million points; the slider would feel awful
    # without this, since every change re-requests the same file.
    app.state.cloud_cache = {}

    def room_or_404(name: str) -> R.RoomPaths:
        room = R.get_room(name, out_dir)
        if room is None:
            raise HTTPException(404, f"no such room: {name}")
        return room

    def load_cloud(room: R.RoomPaths):
        key = str(room.ply)
        cached = app.state.cloud_cache.get(key)
        stamp = room.ply.stat().st_mtime if room.ply.exists() else 0
        if cached and cached[0] == stamp:
            return cached[1], cached[2]

        if not room.ply.exists():
            raise HTTPException(404, "no scene.ply for this room")
        points, colors = R.read_ply(room.ply)
        app.state.cloud_cache[key] = (stamp, points, colors)
        return points, colors

    # --- rooms -------------------------------------------------------------

    @app.get("/api/rooms")
    def list_rooms():
        return {"rooms": [R.room_summary(r) for r in R.discover_rooms(out_dir)]}

    @app.get("/api/rooms/{name}")
    def room_detail(name: str):
        return R.room_summary(room_or_404(name))

    @app.get("/api/rooms/{name}/objects")
    def get_objects(name: str):
        room = room_or_404(name)
        doc = R.load_json(room.objects)
        if doc is None:
            raise HTTPException(404, "no objects.json for this room")
        return doc

    @app.get("/api/rooms/{name}/observations")
    def get_observations(name: str):
        room = room_or_404(name)
        doc = R.load_json(room.observations)
        if doc is None:
            raise HTTPException(
                404,
                "no observations.json — this room was labeled before observations "
                "were persisted. Re-run the label stage, or backfill with "
                "scripts/backfill_observations.py",
            )
        return doc

    @app.get("/api/rooms/{name}/trajectory")
    def get_trajectory(name: str):
        room = room_or_404(name)
        if not room.trajectory.exists():
            raise HTTPException(404, "no trajectory.txt for this room")
        poses = R.read_trajectory(room.trajectory)
        return {"poses": poses, "stats": R.trajectory_stats(poses)}

    @app.get("/api/rooms/{name}/frames")
    def list_frames(name: str):
        room = room_or_404(name)
        if not room.frames_dir.is_dir():
            return {"frames": []}
        names = sorted(p.name for p in room.frames_dir.glob("*.png"))
        return {"frames": [{"index": i, "name": n} for i, n in enumerate(names)]}

    @app.get("/api/rooms/{name}/frames/{index}")
    def get_frame(name: str, index: int):
        room = room_or_404(name)
        files = sorted(room.frames_dir.glob("*.png")) if room.frames_dir.is_dir() else []
        if not 0 <= index < len(files):
            raise HTTPException(404, f"no frame {index}")
        return FileResponse(files[index], media_type="image/png")

    @app.get("/api/rooms/{name}/cloud")
    def get_cloud(name: str, max_points: int = Query(300_000, ge=1000, le=5_000_000)):
        room = room_or_404(name)
        points, colors = load_cloud(room)
        points, colors = R.decimate(points, colors, max_points)
        return Response(
            content=R.pack_cloud(points, colors),
            media_type="application/octet-stream",
            headers={"X-Point-Count": str(len(points))},
        )

    # --- floor plan --------------------------------------------------------

    @app.get("/api/rooms/{name}/floorplan/meta")
    def floorplan_meta(name: str, up: str | None = None, size: int = Query(900, ge=200, le=2000)):
        room = room_or_404(name)
        points, _ = load_cloud(room)

        report = fp.up_report(points, R.pose_matrices(room.trajectory))
        axis = report["axis"]
        if up and up.lower() in fp.AXES:
            axis = fp.AXES[up.lower()]
            report = {**report, "axis": axis, "confidence": 1.0, "source": "manual",
                      "notes": [f"up axis overridden to {up.lower()} in the UI"]}

        transform = fp.build_transform(points, axis, size=size)
        objects = (R.load_json(room.objects) or {}).get("objects", [])

        return {
            "transform": transform.as_dict(),
            "up_estimate": {**report, "axis": "xyz"[axis]},
            "level": load_level_record(room.root),
            "footprints": fp.object_footprints(objects, transform),
        }

    @app.get("/api/rooms/{name}/floorplan.png")
    def floorplan_png(
        name: str,
        up: str | None = None,
        size: int = Query(900, ge=200, le=2000),
        drop_ceiling: float = Query(0.15, ge=0.0, le=0.9),
    ):
        from PIL import Image

        room = room_or_404(name)
        points, colors = load_cloud(room)

        axis = fp.up_report(points, R.pose_matrices(room.trajectory))["axis"]
        if up and up.lower() in fp.AXES:
            axis = fp.AXES[up.lower()]

        transform = fp.build_transform(points, axis, size=size)
        image = fp.render_plan(points, colors, transform, drop_ceiling=drop_ceiling)

        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    # --- re-fusion ---------------------------------------------------------

    @app.post("/api/rooms/{name}/refuse")
    def post_refuse(name: str, req: RefuseRequest):
        room = room_or_404(name)
        if not room.observations.exists():
            raise HTTPException(404, "no observations.json — cannot re-fuse")

        obs_set = load_observations(room.observations)
        up, floor_height = scene_frame_of(room.root)
        objects = refuse(
            obs_set,
            radius_floor=req.radius_floor,
            radius_scale=req.radius_scale,
            min_obb_iou=req.min_obb_iou,
            min_confidence=req.min_confidence,
            min_observations=req.min_observations,
            up=up,
            floor_height=floor_height,
        )

        previous = R.load_json(room.objects) or {}
        doc = objects_doc(
            name, objects,
            scale_verified=previous.get("scale_verified", False),
            n_frames_labeled=len(obs_set.frames_labeled),
        )
        if req.save:
            save_objects(room.objects, doc)

        return {
            **doc,
            "saved": req.save,
            "n_observations": obs_set.n,
            "previous_count": len(previous.get("objects", [])),
            "radius": radius_summary(obs_set, req.radius_floor, req.radius_scale),
        }

    # --- runs --------------------------------------------------------------

    @app.get("/api/videos")
    def list_videos():
        found = []
        for folder in ("video_examples", "data/rooms", "data"):
            d = project_root / folder
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in VIDEO_SUFFIXES:
                    found.append(
                        {
                            "path": str(p.relative_to(project_root)).replace("\\", "/"),
                            "name": p.name,
                            "mb": round(p.stat().st_size / 1e6, 1),
                        }
                    )
        return {"videos": found}

    @app.post("/api/runs")
    def start_run(req: RunRequest):
        source = (project_root / req.source).resolve()
        if not str(source).startswith(str(project_root.resolve())):
            raise HTTPException(400, "source must be inside the project directory")
        if not source.exists():
            raise HTTPException(404, f"no such file: {req.source}")

        room = req.room.strip()
        if not room or "/" in room or "\\" in room or room.startswith("."):
            raise HTTPException(400, "invalid room name")

        try:
            run = app.state.runs.start(
                req.source, room,
                n_frames=req.n_frames, direct=req.direct,
                scene_graph=req.scene_graph, allow_mixed=req.allow_mixed,
                no_label=req.no_label,
            )
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

        return run.snapshot()

    @app.get("/api/runs")
    def list_runs():
        active = app.state.runs.active
        return {"recent": app.state.runs.recent(), "active": active.id if active else None}

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str, since: int = 0):
        run = app.state.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run.snapshot(since=since)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        if not app.state.runs.cancel(run_id):
            raise HTTPException(409, "run is not cancellable")
        return {"cancelled": True}

    # --- static ------------------------------------------------------------

    @app.get("/")
    def index():
        page = STATIC / "index.html"
        if not page.exists():
            return JSONResponse({"error": "frontend not built"}, status_code=500)
        return FileResponse(page, media_type="text/html")

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, out_dir: str = "out",
          open_browser: bool = False) -> None:
    import uvicorn

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()

    uvicorn.run(create_app(out_dir=out_dir), host=host, port=port, log_level="info")
