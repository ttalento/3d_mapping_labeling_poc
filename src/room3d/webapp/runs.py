"""Background pipeline runs, with progress parsed from the CLI's own output.

Reconstruction takes 12-40 minutes, so an HTTP request cannot hold one. Each run
is a subprocess whose stdout is tailed into a bounded buffer that the UI polls
or streams.

Progress comes from the stage markers the CLI already prints -- `[extract]`,
`[reconstruct]`, `[detect]`, `[cluster]`, `[label]`. Those exist because the CLI
is meant to be watched in a terminal; reusing them means the web UI needs no
special logging mode, and the two views of a run cannot drift apart.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

# Ordered, so a marker also tells us how far through the run we are.
STAGES = ("extract", "reconstruct", "detect", "project", "cluster", "label")
_MARKER = re.compile(r"^\s*\[(extract|reconstruct|detect|project|cluster|label|semantics|frames)\]")

# dust3r's tqdm bars would swamp the log; keep only the occasional heartbeat.
_PROGRESS_BAR = re.compile(r"\d+%\|")

MAX_LINES = 2000


@dataclass
class Run:
    id: str
    room: str
    command: list[str]
    status: str = "running"          # running | done | failed | cancelled
    stage: str = "starting"
    returncode: int | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LINES))
    _process: subprocess.Popen | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            lines = list(self.lines)
        return {
            "id": self.id,
            "room": self.room,
            "status": self.status,
            "stage": self.stage,
            "returncode": self.returncode,
            "elapsed": round(self.elapsed, 1),
            "stage_index": STAGES.index(self.stage) if self.stage in STAGES else -1,
            "n_stages": len(STAGES),
            "total_lines": len(lines),
            "lines": lines[since:],
        }


class RunManager:
    """One run at a time. Concurrent reconstructions would just thrash the CPU."""

    def __init__(self, cwd: Path | None = None):
        self.cwd = Path(cwd or Path.cwd())
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    @property
    def active(self) -> Run | None:
        with self._lock:
            return next((r for r in self._runs.values() if r.status == "running"), None)

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def recent(self, limit: int = 10) -> list[dict]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: -r.started)[:limit]
        return [
            {"id": r.id, "room": r.room, "status": r.status, "stage": r.stage,
             "elapsed": round(r.elapsed, 1)}
            for r in runs
        ]

    def start(
        self,
        source: str,
        room: str,
        *,
        n_frames: int | None = None,
        direct: bool = True,
        scene_graph: str | None = None,
        allow_mixed: bool = False,
        no_label: bool = False,
    ) -> Run:
        if self.active is not None:
            raise RuntimeError("a run is already in progress")

        # Invoke the module rather than the console script: the console script
        # may not be on PATH when the server was started from a venv-aware
        # launcher, and `-u` is what makes progress appear live.
        cmd = [sys.executable, "-u", "-m", "room3d.cli", "run", source, "--room", room]
        if n_frames:
            cmd += ["--n-frames", str(n_frames)]
        if direct:
            cmd += ["--direct"]
        if scene_graph:
            cmd += ["--scene-graph", scene_graph]
        if allow_mixed:
            cmd += ["--allow-mixed"]
        if no_label:
            cmd += ["--no-label"]

        run = Run(id=uuid.uuid4().hex[:12], room=room, command=cmd)
        with self._lock:
            self._runs[run.id] = run

        threading.Thread(target=self._pump, args=(run,), daemon=True).start()
        return run

    def cancel(self, run_id: str) -> bool:
        run = self.get(run_id)
        if not run or run.status != "running" or run._process is None:
            return False
        run._process.terminate()
        run.status = "cancelled"
        return True

    def _pump(self, run: Run) -> None:
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        try:
            import os

            proc = subprocess.Popen(
                run.command,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env={**os.environ, **env},
            )
        except OSError as exc:
            run.status, run.returncode = "failed", -1
            run.finished = time.time()
            self._append(run, f"failed to start: {exc}")
            return

        run._process = proc
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.rstrip("\n")
            if _PROGRESS_BAR.search(line):
                continue
            self._append(run, line)

            marker = _MARKER.match(line)
            if marker:
                stage = marker.group(1)
                run.stage = stage if stage in STAGES else run.stage

        proc.wait()
        run.returncode = proc.returncode
        run.finished = time.time()
        if run.status == "running":
            run.status = "done" if proc.returncode == 0 else "failed"
        run.stage = "label" if run.status == "done" else run.stage
        self._append(run, f"--- {run.status} (exit {proc.returncode}) in {run.elapsed:.0f}s ---")

    @staticmethod
    def _append(run: Run, line: str) -> None:
        with run._lock:
            run.lines.append(line)
