"""Stage 1: pull a small, sharp, non-redundant frame set out of a walkthrough.

Frame count is the dominant cost knob for everything downstream, so the goal is
not "many frames" but "the fewest frames that still cover the room". Two filters
earn their place on handheld phone video:

  * blur rejection -- a walkthrough is full of motion blur, and a blurred frame
    poisons the reconstruction it takes part in
  * near-duplicate rejection -- standing still produces many identical frames
    that cost inference time and contribute no new geometry
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class ExtractedFrame:
    index: int
    timestamp: float
    sharpness: float
    path: Path


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian. Higher is sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _thumbnail(image: np.ndarray, size: int = 32) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    norm = np.linalg.norm(small)
    return small / norm if norm > 1e-6 else small


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip((a * b).sum(), -1.0, 1.0))


_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def video_rotation(path: Path) -> int:
    """Rotation, in degrees, that the container says to apply on playback.

    Phones shoot portrait but store the frames landscape with a rotate flag.
    OpenCV hands back the raw, unrotated frames, so without this every frame is
    on its side -- which costs little geometrically but a great deal at the VLM,
    which is being asked to name furniture in a picture lying on its side.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    except AttributeError:          # OpenCV < 4.5
        return 0
    finally:
        cap.release()

    return int(meta) % 360 if meta else 0


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    code = _ROTATIONS.get(rotation)
    return cv2.rotate(frame, code) if code is not None else frame


def _read_video_frames(path: Path) -> list[tuple[float, np.ndarray]]:
    """Decode a video to (timestamp, BGR frame) pairs, upright."""
    rotation = video_rotation(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[tuple[float, np.ndarray]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((idx / fps, _apply_rotation(frame, rotation)))
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")

    if rotation:
        print(f"[extract] applied {rotation}deg rotation from video metadata")
    return frames


def _read_image_folder(path: Path) -> list[tuple[float, np.ndarray]]:
    paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise RuntimeError(f"no images found in {path}")

    out = []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"could not read image: {p}")
        out.append((float(i), img))
    return out


def select_frames(
    candidates: list[tuple[float, np.ndarray]],
    n_frames: int,
    *,
    blur_percentile: float = 25.0,
    dedup_correlation: float = 0.985,
    min_frames: int = 4,
) -> list[int]:
    """Choose which candidate indices to keep. Pure, so it is testable.

    Walks the candidates in temporal order, taking a uniform stride and, within
    each stride window, the sharpest frame that is not a near-duplicate of the
    one already kept.
    """
    if len(candidates) <= min_frames:
        return list(range(len(candidates)))

    sharp = np.array([sharpness(img) for _, img in candidates])
    floor = float(np.percentile(sharp, blur_percentile))

    stride = max(1, len(candidates) // n_frames)
    kept: list[int] = []
    last_thumb: np.ndarray | None = None

    for start in range(0, len(candidates), stride):
        window = range(start, min(start + stride, len(candidates)))

        # Prefer sharp frames, but never let the blur floor empty a window.
        usable = [i for i in window if sharp[i] >= floor] or list(window)
        order = sorted(usable, key=lambda i: -sharp[i])

        for i in order:
            thumb = _thumbnail(candidates[i][1])
            if last_thumb is not None and _correlation(thumb, last_thumb) > dedup_correlation:
                continue
            kept.append(i)
            last_thumb = thumb
            break

        if len(kept) >= n_frames:
            break

    # Dedup can starve the set; backfill with the sharpest unused frames.
    if len(kept) < min_frames:
        spare = sorted(set(range(len(candidates))) - set(kept), key=lambda i: -sharp[i])
        kept = sorted(kept + spare[: min_frames - len(kept)])

    return kept


def extract_frames(
    source: str | Path,
    out_dir: str | Path,
    n_frames: int = 24,
    *,
    blur_percentile: float = 25.0,
    dedup_correlation: float = 0.985,
    min_frames: int = 4,
) -> list[ExtractedFrame]:
    """Decode `source` (video file or image folder), select frames, write PNGs."""
    source, out_dir = Path(source), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        candidates = _read_image_folder(source)
    elif source.suffix.lower() in VIDEO_SUFFIXES:
        candidates = _read_video_frames(source)
    else:
        raise ValueError(f"unsupported source: {source} (expected a video or a folder)")

    keep = select_frames(
        candidates,
        n_frames,
        blur_percentile=blur_percentile,
        dedup_correlation=dedup_correlation,
        min_frames=min_frames,
    )

    extracted: list[ExtractedFrame] = []
    for out_idx, cand_idx in enumerate(keep):
        timestamp, image = candidates[cand_idx]
        path = out_dir / f"{out_idx:03d}.png"
        cv2.imwrite(str(path), image)
        extracted.append(
            ExtractedFrame(
                index=out_idx,
                timestamp=float(timestamp),
                sharpness=sharpness(image),
                path=path,
            )
        )
    return extracted
