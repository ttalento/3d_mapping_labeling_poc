"""Discover and load the artifacts of completed runs under out/.

Everything here tolerates a partially-complete room. A run that reconstructed
but died during labeling is exactly the state you most want to look at, so a
missing objects.json must degrade the response, not fail it.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARTIFACTS = ("scene.ply", "trajectory.txt", "frames.npz", "objects.json", "observations.json")


@dataclass
class RoomPaths:
    name: str
    root: Path

    @property
    def ply(self) -> Path:
        return self.root / "scene.ply"

    @property
    def trajectory(self) -> Path:
        return self.root / "trajectory.txt"

    @property
    def npz(self) -> Path:
        return self.root / "frames.npz"

    @property
    def objects(self) -> Path:
        return self.root / "objects.json"

    @property
    def observations(self) -> Path:
        return self.root / "observations.json"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    def has(self) -> dict[str, bool]:
        return {name: (self.root / name).exists() for name in ARTIFACTS}


def discover_rooms(out_dir: str | Path = "out") -> list[RoomPaths]:
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    return [
        RoomPaths(name=d.name, root=d)
        for d in sorted(out_dir.iterdir())
        if d.is_dir() and any((d / a).exists() for a in ARTIFACTS)
    ]


def get_room(name: str, out_dir: str | Path = "out") -> RoomPaths | None:
    # Reject traversal explicitly rather than relying on the router: this is the
    # only place a user-supplied string becomes a filesystem path.
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    root = Path(out_dir) / name
    return RoomPaths(name=name, root=root) if root.is_dir() else None


# --- PLY -------------------------------------------------------------------


def read_ply_header(path: Path) -> tuple[int, int]:
    """Return (vertex_count, header_byte_length) for our binary PLY writer."""
    with open(path, "rb") as f:
        blob = f.read(2048)
    end = blob.find(b"end_header\n")
    if end == -1:
        raise ValueError(f"not a PLY we recognise: {path}")

    header = blob[:end].decode("ascii", errors="replace")
    count = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
    return count, end + len(b"end_header\n")


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the xyz+rgb PLY written by artifacts.save_ply."""
    path = Path(path)
    count, offset = read_ply_header(path)

    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    rows = np.fromfile(path, dtype=dtype, count=count, offset=offset)

    points = np.stack([rows["x"], rows["y"], rows["z"]], axis=1)
    colors = np.stack([rows["red"], rows["green"], rows["blue"]], axis=1)
    return points, colors


def decimate(
    points: np.ndarray, colors: np.ndarray, max_points: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random subsample, seeded so the view does not shimmer on reload."""
    n = len(points)
    if max_points <= 0 or n <= max_points:
        return points, colors
    idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
    idx.sort()                      # keep memory access roughly sequential
    return points[idx], colors[idx]


def pack_cloud(points: np.ndarray, colors: np.ndarray) -> bytes:
    """Binary framing: [u32 count][float32 xyz * 3n][uint8 rgb * 3n].

    Shipping the PLY and parsing it in JS costs seconds on a million points;
    this is two typed-array views over one ArrayBuffer.
    """
    n = len(points)
    return (
        struct.pack("<I", n)
        + np.ascontiguousarray(points, dtype="<f4").tobytes()
        + np.ascontiguousarray(colors, dtype=np.uint8).tobytes()
    )


# --- trajectory ------------------------------------------------------------


def read_trajectory(path: str | Path) -> list[dict]:
    """Parse TUM `t x y z qx qy qz qw` into per-frame records."""
    lines = Path(path).read_text().strip().splitlines()
    poses = []
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 8:
            continue
        poses.append(
            {
                "index": i,
                "t": float(parts[0]),
                "position": [float(v) for v in parts[1:4]],
                "quaternion": [float(v) for v in parts[4:8]],
            }
        )

    # Step distances make a pan-versus-walk immediately legible.
    for i, p in enumerate(poses):
        if i == 0:
            p["step"] = 0.0
        else:
            a = np.asarray(poses[i - 1]["position"])
            b = np.asarray(p["position"])
            p["step"] = round(float(np.linalg.norm(b - a)), 4)
    return poses


def pose_matrices(path: str | Path) -> np.ndarray | None:
    """Read trajectory.txt as (N, 4, 4) camera-to-world matrices.

    The up-axis estimator needs orientations, and trajectory.txt carries them at
    a few hundred bytes. Reading them out of frames.npz instead would mean
    decompressing hundreds of megabytes of pointmaps to get at eight rotations.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        records = read_trajectory(path)
    except (OSError, ValueError):
        return None
    if not records:
        return None

    out = np.zeros((len(records), 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0
    for i, rec in enumerate(records):
        out[i, :3, :3] = _quat_to_rotmat(rec["quaternion"])
        out[i, :3, 3] = rec["position"]
    return out


def _quat_to_rotmat(q: list[float]) -> np.ndarray:
    """(qx, qy, qz, qw) -> 3x3. Inverse of artifacts._rotmat_to_quat."""
    x, y, z, w = (float(v) for v in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def trajectory_stats(poses: list[dict]) -> dict:
    """Camera span and path length -- the numbers that explain a bad fusion."""
    if not poses:
        return {"n_poses": 0, "span": [0, 0, 0], "span_max": 0.0, "path_length": 0.0}

    xyz = np.asarray([p["position"] for p in poses])
    span = xyz.max(axis=0) - xyz.min(axis=0)
    return {
        "n_poses": len(poses),
        "span": [round(float(v), 4) for v in span],
        "span_max": round(float(span.max()), 4),
        "path_length": round(float(sum(p["step"] for p in poses)), 4),
        "mean_step": round(float(np.mean([p["step"] for p in poses[1:]])), 4)
        if len(poses) > 1 else 0.0,
    }


# --- summaries -------------------------------------------------------------


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def room_summary(room: RoomPaths) -> dict:
    """Everything the UI needs to render a room card, tolerating missing pieces."""
    has = room.has()
    summary: dict = {"name": room.name, "artifacts": has}

    if has["scene.ply"]:
        try:
            summary["n_points"], _ = read_ply_header(room.ply)
        except (OSError, ValueError):
            summary["n_points"] = 0

    frames = sorted(room.frames_dir.glob("*.png")) if room.frames_dir.is_dir() else []
    summary["n_frames"] = len(frames)

    if has["trajectory.txt"]:
        try:
            summary["trajectory"] = trajectory_stats(read_trajectory(room.trajectory))
        except OSError:
            pass

    objects = load_json(room.objects) if has["objects.json"] else None
    if objects:
        summary["n_objects"] = len(objects.get("objects", []))
        summary["scale_verified"] = objects.get("scale_verified", False)
        summary["units"] = objects.get("units", "meters")

    observations = load_json(room.observations) if has["observations.json"] else None
    if observations:
        summary["n_observations"] = len(observations.get("observations", []))

    return summary
