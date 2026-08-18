"""Reading and writing the artifacts that connect the pipeline stages.

`frames.npz` is the important one: it is the documented interface between the
reconstruction half and the labeling half. Any reconstructor that can produce a
per-frame world-frame pointmap plus a pose (MASt3R-SLAM keyframes, VGGT, DUSt3R
global alignment) can feed the labeling pipeline by writing this file.

Layout of frames.npz
--------------------
    images       (N, H, W, 3) uint8    the exact pixels the reconstructor saw
    pts3d        (N, H, W, 3) float32  per-pixel 3D point, WORLD frame
    conf_mask    (N, H, W)    bool     per-pixel geometry confidence
    poses        (N, 4, 4)    float32  camera-to-world
    intrinsics   (N, 3, 3)    float32  pinhole K for the (H, W) images above
    frame_ids    (N,)         int32    index back into the extracted frame set
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Reconstruction:
    images: np.ndarray
    pts3d: np.ndarray
    conf_mask: np.ndarray
    poses: np.ndarray
    intrinsics: np.ndarray
    frame_ids: np.ndarray

    def __post_init__(self) -> None:
        n, h, w = self.images.shape[:3]
        expect = {
            "images": (n, h, w, 3),
            "pts3d": (n, h, w, 3),
            "conf_mask": (n, h, w),
            "poses": (n, 4, 4),
            "intrinsics": (n, 3, 3),
            "frame_ids": (n,),
        }
        for name, shape in expect.items():
            got = getattr(self, name).shape
            if got != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {got}")

    @property
    def n_frames(self) -> int:
        return self.images.shape[0]

    @property
    def image_hw(self) -> tuple[int, int]:
        return self.images.shape[1], self.images.shape[2]


def save_frames_npz(path: str | Path, recon: Reconstruction) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=recon.images,
        pts3d=recon.pts3d.astype(np.float32),
        conf_mask=recon.conf_mask,
        poses=recon.poses.astype(np.float32),
        intrinsics=recon.intrinsics.astype(np.float32),
        frame_ids=recon.frame_ids.astype(np.int32),
    )


def load_frames_npz(path: str | Path) -> Reconstruction:
    with np.load(path) as d:
        return Reconstruction(
            images=d["images"],
            pts3d=d["pts3d"],
            conf_mask=d["conf_mask"],
            poses=d["poses"],
            intrinsics=d["intrinsics"],
            frame_ids=d["frame_ids"],
        )


# The 3D points behind each observation, keyed "obs_<index>" into the
# observations.json list. Kept out of the JSON because they are three orders of
# magnitude larger than everything else in it and a human reads that file.
OBSERVATION_POINTS_NAME = "observation_points.npz"


def save_observation_points(obs_path: str | Path, observations) -> Path | None:
    """Write each observation's 3D points beside `observations.json`.

    Re-fusing in the viewer has to produce the same box the pipeline did, and a
    box can only be fit to points -- so the points have to outlive the run.
    Returns None when no observation carries any.
    """
    payload = {
        f"obs_{i}": np.asarray(obs.points, dtype=np.float32)
        for i, obs in enumerate(observations)
        if getattr(obs, "points", None) is not None and len(obs.points)
    }
    if not payload:
        return None

    path = Path(obs_path).with_name(OBSERVATION_POINTS_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_observation_points(obs_path: str | Path) -> dict[int, np.ndarray]:
    """Read the sidecar back, keyed by observation index. Missing file -> empty."""
    path = Path(obs_path).with_name(OBSERVATION_POINTS_NAME)
    if not path.exists():
        return {}
    with np.load(path) as d:
        return {int(k.split("_")[1]): np.asarray(d[k], dtype=np.float64) for k in d.files}


def save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary little-endian PLY. colors are uint8 RGB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError(f"points/colors length mismatch: {len(points)} vs {len(colors)}")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    rows = np.empty(len(points), dtype=dtype)
    rows["x"], rows["y"], rows["z"] = points.T
    rows["red"], rows["green"], rows["blue"] = colors.T

    with open(path, "wb") as f:
        f.write(header)
        f.write(rows.tobytes())


def save_trajectory_tum(
    path: str | Path, poses: np.ndarray, timestamps: np.ndarray | None = None
) -> None:
    """Write camera-to-world poses in TUM format: `t x y z qx qy qz qw`.

    Matches MASt3R-SLAM's trajectory format so downstream tooling is
    interchangeable between the two reconstruction backends.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.asarray(poses, dtype=np.float64)
    if timestamps is None:
        timestamps = np.arange(len(poses), dtype=np.float64)

    lines = []
    for t, T in zip(timestamps, poses):
        x, y, z = T[:3, 3]
        qx, qy, qz, qw = _rotmat_to_quat(T[:3, :3])
        lines.append(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}")
    Path(path).write_text("\n".join(lines) + "\n")


def _rotmat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (qx, qy, qz, qw), using the numerically stable branch."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw, qx, qy, qz = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw, qx, qy, qz = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw, qx, qy, qz = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw, qx, qy, qz = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s

    return float(qx), float(qy), float(qz), float(qw)
