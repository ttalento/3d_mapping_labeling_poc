"""Stage 4: look at the result.

The fastest way to tell whether labels landed on the right objects is to see the
boxes sitting on the cloud. Gate 7 is an eyeball test and this is the eyeball.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Distinct hues, stable per object index so colours do not shuffle between runs.
_PALETTE = np.array([
    [0.90, 0.29, 0.23], [0.18, 0.60, 0.86], [0.15, 0.68, 0.38],
    [0.95, 0.61, 0.07], [0.61, 0.35, 0.71], [0.90, 0.49, 0.13],
    [0.10, 0.74, 0.61], [0.83, 0.33, 0.60], [0.20, 0.29, 0.37],
    [0.95, 0.77, 0.06],
])


def _obb_lineset(obj: dict, colour: np.ndarray):
    import open3d as o3d

    obb = obj["obb"]
    extent = np.asarray(obb["extent"], dtype=float)
    if not np.all(np.isfinite(extent)) or np.all(extent <= 1e-6):
        return None

    box = o3d.geometry.OrientedBoundingBox(
        center=np.asarray(obb["center"], dtype=float),
        R=np.asarray(obb["R"], dtype=float),
        extent=np.maximum(extent, 1e-3),
    )
    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    lines.paint_uniform_color(colour)
    return lines


def _camera_frustums(trajectory_path: Path, scale: float = 0.12):
    """Draw the recovered trajectory so the poses are visible, not just implied."""
    import open3d as o3d

    if not trajectory_path.exists():
        return []

    centres = []
    for line in trajectory_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            centres.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(centres) < 2:
        return []

    centres = np.asarray(centres)
    geometries = []

    path = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centres),
        lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centres) - 1)]),
    )
    path.paint_uniform_color([0.1, 0.1, 0.1])
    geometries.append(path)

    for c in centres:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.25)
        marker.translate(c)
        marker.paint_uniform_color([0.05, 0.05, 0.05])
        geometries.append(marker)

    return geometries


def view(
    ply_path: str | Path,
    objects_path: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    *,
    min_confidence: float = 0.0,
    show_trajectory: bool = True,
) -> None:
    import open3d as o3d

    ply_path = Path(ply_path)
    cloud = o3d.io.read_point_cloud(str(ply_path))
    if cloud.is_empty():
        raise RuntimeError(f"point cloud is empty: {ply_path}")

    geometries = [cloud]
    print(f"[view] {len(cloud.points)} points from {ply_path.name}")

    if objects_path and Path(objects_path).exists():
        data = json.loads(Path(objects_path).read_text())
        objects = [
            o for o in data.get("objects", [])
            if o.get("confidence", 0.0) >= min_confidence
        ]
        for i, obj in enumerate(objects):
            colour = _PALETTE[i % len(_PALETTE)]
            lines = _obb_lineset(obj, colour)
            if lines is not None:
                geometries.append(lines)

            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            marker.translate(np.asarray(obj["centroid"], dtype=float))
            marker.paint_uniform_color(colour)
            geometries.append(marker)

        print(f"[view] {len(objects)} objects (confidence >= {min_confidence})")
        for obj in objects:
            print(f"       {obj['label']:<20} {obj['confidence']:.2f}  "
                  f"{np.round(obj['centroid'], 2).tolist()}")

    if show_trajectory and trajectory_path:
        geometries.extend(_camera_frustums(Path(trajectory_path)))

    o3d.visualization.draw_geometries(
        geometries, window_name=f"room3d — {ply_path.parent.name}", width=1400, height=900
    )
