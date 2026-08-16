"""Put the reconstruction the right way up.

DUSt3R's global aligner has no gravity vector. It anchors the world frame on one
reference camera, so "world" is really "camera N's frame" -- and in the OpenCV
convention that frame's +Y axis points *down* the image. A phone held upright
therefore produces a scene whose +Y is roughly gravity-down, which renders
upside down in any viewer that assumes +Y up (three.js does) and defeats any
floor detection that looks for a flat axis.

Two independent signals recover gravity, and using both is what makes the answer
trustworthy rather than a guess:

1. **The camera poses.** Each pose's second column is the camera's own "down"
   direction expressed in world coordinates. A handheld phone is roughly upright
   throughout, so averaging those gives gravity directly. This needs no scene
   structure at all, which matters because it still works on a reconstruction too
   sparse or warped to find a floor in.
2. **The floor plane.** RANSAC over the lowest slice of the cloud, constrained to
   stay near the pose estimate so a wall cannot win. This corrects the systematic
   error the poses cannot see: if you held the phone tilted down the whole time,
   the mean camera down-vector is tilted down too.

When the two agree, the estimate is solid. When they disagree, the confidence
score says so rather than silently picking one.

The transform this produces is *rigid* -- a rotation and a translation. It moves
nothing relative to anything else, so distances, OBB extents and fusion results
are all invariant. What it buys is that +Y means up, y=0 means the floor, and
"how high off the ground is this object" becomes a coordinate lookup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Canonical target frame: +Y up, floor at y = 0, scene centred on the origin in
# X/Z. +Y matches three.js's default camera up, which is what the viewer uses.
UP_AXIS = 1
UP_VECTOR = np.array([0.0, 1.0, 0.0])


@dataclass
class UpEstimate:
    """Where "up" is, and how much to believe it."""

    up: np.ndarray                    # unit vector in the current world frame
    floor_offset: float               # the floor plane is {x : up . x == this}
    confidence: float                 # 0-1
    source: str                       # "poses+floor" | "poses" | "extent"
    pose_tilt_deg: float = 0.0        # mean angle of each camera down from their mean
    floor_inlier_frac: float = 0.0    # share of the low slice on the fitted plane
    agreement_deg: float = 0.0        # angle between the pose and floor estimates
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "up": [round(float(v), 6) for v in self.up],
            "floor_offset": round(float(self.floor_offset), 6),
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "pose_tilt_deg": round(float(self.pose_tilt_deg), 2),
            "floor_inlier_frac": round(float(self.floor_inlier_frac), 3),
            "agreement_deg": round(float(self.agreement_deg), 2),
            "notes": list(self.notes),
        }


# --- the two estimators ----------------------------------------------------


def up_from_poses(poses: np.ndarray) -> tuple[np.ndarray, float]:
    """Gravity from the camera orientations. Returns (up, mean tilt in degrees).

    Column 1 of a camera-to-world rotation is the camera's +Y axis in world
    coordinates, and +Y is down the image in the OpenCV convention every model
    in this stack uses. The mean of those, negated, is up.

    The returned tilt is the mean angle between each frame's down-vector and
    their mean -- how consistently the phone was held, which is the only thing
    that makes this estimate believable on its own.
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    if len(poses) == 0:
        raise ValueError("no poses given")

    downs = poses[:, :3, 1]
    norms = np.linalg.norm(downs, axis=1, keepdims=True)
    downs = downs / np.maximum(norms, 1e-12)

    mean = downs.mean(axis=0)
    mag = np.linalg.norm(mean)
    if mag < 1e-6:
        # The frames disagree so completely they cancel. No usable prior.
        return np.array([0.0, -1.0, 0.0]), 90.0
    mean = mean / mag

    tilt = np.degrees(np.arccos(np.clip(downs @ mean, -1.0, 1.0))).mean()
    return -mean, float(tilt)


def fit_floor_plane(
    points: np.ndarray,
    up_prior: np.ndarray,
    *,
    slice_quantile: float = 0.25,
    tolerance: float = 0.03,
    max_tilt_deg: float = 35.0,
    iterations: int = 400,
    seed: int = 0,
) -> tuple[np.ndarray, float, float] | None:
    """RANSAC the floor out of the lowest slice. Returns (normal, offset, inliers).

    Constrained two ways, both load-bearing. Only the lowest `slice_quantile` of
    the cloud along the prior is considered, and candidate planes more than
    `max_tilt_deg` from the prior are rejected outright -- otherwise the biggest
    plane in a room is usually a wall, and a confident fit to a wall is a worse
    failure than no fit at all.

    Returns None when nothing plausible is found, which is a real outcome on a
    cloud with no visible floor.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 100:
        return None

    rng = np.random.default_rng(seed)
    if len(pts) > 120_000:
        pts = pts[rng.choice(len(pts), 120_000, replace=False)]

    up_prior = np.asarray(up_prior, dtype=np.float64)
    up_prior = up_prior / np.linalg.norm(up_prior)

    heights = pts @ up_prior
    low = pts[heights <= np.quantile(heights, slice_quantile)]
    if len(low) < 50:
        return None

    min_cos = np.cos(np.radians(max_tilt_deg))
    best_count, best_normal, best_offset = 0, None, 0.0

    for _ in range(iterations):
        tri = low[rng.choice(len(low), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        mag = np.linalg.norm(normal)
        if mag < 1e-9:
            continue
        normal = normal / mag
        if normal @ up_prior < 0:
            normal = -normal
        if normal @ up_prior < min_cos:
            continue

        offset = float(normal @ tri[0])
        count = int((np.abs(low @ normal - offset) < tolerance).sum())
        if count > best_count:
            best_count, best_normal, best_offset = count, normal, offset

    if best_normal is None or best_count < 0.10 * len(low):
        return None

    # Refit on the inliers: three random points fix a plane far more coarsely
    # than a least-squares fit to every point that agrees with it.
    inliers = low[np.abs(low @ best_normal - best_offset) < tolerance]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[-1]
    if normal @ up_prior < 0:
        normal = -normal
    if normal @ up_prior >= min_cos:
        best_normal = normal
        best_offset = float(normal @ centroid)

    return best_normal, best_offset, best_count / len(low)


def estimate_up(
    points: np.ndarray, poses: np.ndarray | None = None, **floor_kwargs
) -> UpEstimate:
    """Combine the pose prior and the floor fit into one answer with a score."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    notes: list[str] = []

    if poses is None or len(np.asarray(poses).reshape(-1, 4, 4)) == 0:
        return _up_from_extent(pts)

    pose_up, tilt = up_from_poses(poses)
    # 60 degrees of average wobble means the phone was tumbling; below ~15 it
    # was held steadily enough that the mean is meaningful.
    pose_conf = float(np.clip(1.0 - tilt / 60.0, 0.0, 1.0))

    fit = fit_floor_plane(pts, pose_up, **floor_kwargs) if len(pts) >= 100 else None
    if fit is None:
        notes.append("no floor plane found; using camera orientations alone")
        offset = float(np.quantile(pts @ pose_up, 0.01)) if len(pts) else 0.0
        return UpEstimate(
            up=pose_up,
            floor_offset=offset,
            # Unverified by scene structure, so it cannot claim a high score.
            confidence=round(pose_conf * 0.7, 3),
            source="poses",
            pose_tilt_deg=tilt,
            notes=notes,
        )

    normal, offset, inlier_frac = fit
    agreement = float(np.degrees(np.arccos(np.clip(normal @ pose_up, -1.0, 1.0))))

    floor_conf = float(np.clip(inlier_frac / 0.35, 0.0, 1.0))
    agree_conf = float(np.clip(1.0 - agreement / 30.0, 0.0, 1.0))
    # Geometric mean: two independent signals agreeing is the evidence, so any
    # one of them failing has to drag the score down rather than be averaged out.
    confidence = float((pose_conf * floor_conf * agree_conf) ** (1 / 3))

    if agreement > 20.0:
        notes.append(
            f"floor plane and camera orientations disagree by {agreement:.0f} deg"
        )

    return UpEstimate(
        up=normal,
        floor_offset=offset,
        confidence=round(confidence, 3),
        source="poses+floor",
        pose_tilt_deg=tilt,
        floor_inlier_frac=inlier_frac,
        agreement_deg=agreement,
        notes=notes,
    )


def _up_from_extent(points: np.ndarray) -> UpEstimate:
    """Last resort with no poses: a room is short in the vertical direction.

    Weak, and honest about it. A room that is taller than it is deep, or a
    partial reconstruction of one wall, breaks this outright -- which is exactly
    what happened before the pose prior existed.
    """
    if len(points) < 10:
        return UpEstimate(np.array(UP_VECTOR), 0.0, 0.0, "extent",
                          notes=["too few points to estimate"])

    spans = points.max(axis=0) - points.min(axis=0)
    order = np.argsort(spans)
    axis = int(order[0])

    ratio = float(spans[order[0]] / max(spans[order[1]], 1e-9))
    confidence = float(np.clip(1.0 - ratio, 0.0, 1.0))

    values = points[:, axis]
    sign = 1.0 if float((values < np.median(values)).mean()) >= 0.5 else -1.0
    up = np.zeros(3)
    up[axis] = sign

    return UpEstimate(
        up=up,
        floor_offset=float(np.quantile(points @ up, 0.01)),
        confidence=round(confidence * 0.5, 3),
        source="extent",
        notes=["no camera poses available; guessed from the scene's shape alone"],
    )


# --- building and applying the transform -----------------------------------


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector `a` onto unit vector `b`."""
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)

    v = np.cross(a, b)
    c = float(a @ b)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # Antiparallel: any perpendicular axis works, so pick a stable one.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        K = _skew(axis)
        return np.eye(3) + 2.0 * K @ K

    K = _skew(v)
    return np.eye(3) + K + K @ K / (1.0 + c)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def levelling_transform(estimate: UpEstimate, points: np.ndarray) -> np.ndarray:
    """The 4x4 rigid transform that stands the scene upright.

    Rotation is the *minimal* one taking the estimated up onto +Y. Any yaw about
    the vertical would also be valid, and choosing one would mean claiming to
    know which way the walls run; we do not, so we do not rotate about it.

    Translation puts the floor at y=0 and centres X/Z on the cloud's median, so
    an object's y coordinate reads directly as its height off the ground.
    """
    R = rotation_between(estimate.up, UP_VECTOR)

    T = np.eye(4)
    T[:3, :3] = R
    # After rotating, the floor plane's height is unchanged: y' = up . x.
    T[1, 3] = -estimate.floor_offset

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts):
        rotated = transform_points(T, pts)
        centre = np.median(rotated, axis=0)
        T[0, 3] -= centre[0]
        T[2, 3] -= centre[2]

    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (..., 3) points, shape preserved."""
    pts = np.asarray(points, dtype=np.float64)
    shape = pts.shape
    flat = pts.reshape(-1, 3)
    out = flat @ np.asarray(T[:3, :3], dtype=np.float64).T + np.asarray(T[:3, 3])
    return out.reshape(shape)


def transform_poses(T: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Compose a world transform onto camera-to-world poses."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    return np.asarray(T, dtype=np.float64)[None] @ poses


def transform_obb(T: np.ndarray, obb: dict) -> dict:
    """Rotate an oriented box. Extents are invariant under a rigid transform."""
    out = dict(obb)
    R = np.asarray(T[:3, :3], dtype=np.float64)
    if "center" in obb:
        out["center"] = transform_points(T, np.asarray(obb["center"], float)).tolist()
    if "R" in obb:
        out["R"] = (R @ np.asarray(obb["R"], dtype=np.float64)).tolist()
    return out


def transform_records(T: np.ndarray, records: list[dict]) -> list[dict]:
    """Move every 3D field in a list of object/observation dicts."""
    out = []
    for rec in records:
        rec = dict(rec)
        if rec.get("centroid") is not None:
            rec["centroid"] = transform_points(
                T, np.asarray(rec["centroid"], float)
            ).round(4).tolist()
        if isinstance(rec.get("obb"), dict):
            rec["obb"] = transform_obb(T, rec["obb"])
        out.append(rec)
    return out


# --- applying it to a whole room on disk ------------------------------------


@dataclass
class LevelResult:
    transform: np.ndarray
    estimate: UpEstimate
    rotation_deg: float
    files: list[str]


def level_reconstruction(recon, estimate: UpEstimate | None = None):
    """Return a levelled copy of a `Reconstruction`, plus the transform used."""
    from .artifacts import Reconstruction

    points = recon.pts3d[recon.conf_mask]
    if estimate is None:
        estimate = estimate_up(points, recon.poses)

    T = levelling_transform(estimate, points)
    levelled = Reconstruction(
        images=recon.images,
        pts3d=transform_points(T, recon.pts3d).astype(np.float32),
        conf_mask=recon.conf_mask,
        poses=transform_poses(T, recon.poses).astype(np.float32),
        intrinsics=recon.intrinsics,
        frame_ids=recon.frame_ids,
    )
    return levelled, T, estimate


def level_room(out_dir: str | Path, *, up: str | None = None, verbose: bool = True) -> LevelResult:
    """Rewrite a room's artifacts so +Y is up and the floor sits at y = 0.

    Every 3D artifact moves together -- cloud, poses, object boxes, observation
    centroids -- because a half-levelled room is worse than an upside-down one.
    The applied transform is recorded in `level.json`, so this is auditable and
    exactly invertible; running it twice is a near no-op rather than a double
    rotation, because re-estimating an already-level scene returns +Y.
    """
    from .artifacts import (
        load_frames_npz,
        save_frames_npz,
        save_ply,
        save_trajectory_tum,
    )

    out_dir = Path(out_dir)
    npz = out_dir / "frames.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found; reconstruct this room first")

    recon = load_frames_npz(npz)
    points = recon.pts3d[recon.conf_mask]

    if up:
        estimate = _forced_estimate(up, points)
    else:
        estimate = estimate_up(points, recon.poses)

    levelled, T, estimate = level_reconstruction(recon, estimate)
    angle = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))

    written: list[str] = []

    save_frames_npz(npz, levelled)
    written.append(npz.name)

    save_ply(
        out_dir / "scene.ply",
        levelled.pts3d[levelled.conf_mask],
        levelled.images[levelled.conf_mask],
    )
    written.append("scene.ply")

    save_trajectory_tum(out_dir / "trajectory.txt", levelled.poses)
    written.append("trajectory.txt")

    for name, key in (("objects.json", "objects"), ("observations.json", "observations")):
        path = out_dir / name
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        doc[key] = transform_records(T, doc.get(key, []))
        doc["frame"] = "y_up_floor_at_zero"
        path.write_text(json.dumps(doc, indent=2))
        written.append(name)

    record = out_dir / "level.json"
    previous = json.loads(record.read_text())["cumulative"] if record.exists() else np.eye(4).tolist()
    cumulative = (T @ np.asarray(previous, dtype=np.float64)).tolist()
    record.write_text(
        json.dumps(
            {
                "convention": "y_up_floor_at_zero",
                "applied": T.tolist(),
                "cumulative": cumulative,
                "rotation_deg": round(angle, 2),
                "estimate": estimate.as_dict(),
            },
            indent=2,
        )
    )
    written.append("level.json")

    if verbose:
        print(f"[level] up estimate: {np.round(estimate.up, 3).tolist()} "
              f"({estimate.source}, confidence {estimate.confidence:.2f})")
        print(f"[level] rotated scene by {angle:.1f} deg; floor -> y=0")
        for note in estimate.notes:
            print(f"[level] note: {note}")
        print(f"[level] rewrote {', '.join(written)}")

    return LevelResult(transform=T, estimate=estimate, rotation_deg=angle, files=written)


def _forced_estimate(up: str, points: np.ndarray) -> UpEstimate:
    """Manual override: `--up -y` etc., for when both estimators are wrong."""
    text = up.strip().lower()
    sign = -1.0 if text.startswith("-") else 1.0
    letter = text.lstrip("+-")
    if letter not in ("x", "y", "z"):
        raise ValueError(f"--up must be one of x, y, z, -x, -y, -z (got {up!r})")

    vector = np.zeros(3)
    vector["xyz".index(letter)] = sign
    offset = float(np.quantile(points @ vector, 0.01)) if len(points) else 0.0
    return UpEstimate(
        up=vector,
        floor_offset=offset,
        confidence=1.0,
        source="manual",
        notes=[f"up axis forced to {text} by the caller"],
    )


def load_level_record(out_dir: str | Path) -> dict | None:
    path = Path(out_dir) / "level.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
