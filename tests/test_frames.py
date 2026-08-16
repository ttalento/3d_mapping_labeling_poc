"""Frame selection: the two filters that matter on handheld phone video."""

import cv2
import numpy as np
import pytest

from room3d.frames import (
    _apply_rotation,
    extract_frames,
    select_frames,
    sharpness,
    video_rotation,
)


def noise_frame(seed: int, size: int = 96) -> np.ndarray:
    """A sharp, high-texture frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def blurred(frame: np.ndarray, k: int = 15) -> np.ndarray:
    return cv2.GaussianBlur(frame, (k, k), 0)


def as_candidates(frames) -> list[tuple[float, np.ndarray]]:
    return [(float(i), f) for i, f in enumerate(frames)]


# --- sharpness ---------------------------------------------------------------


def test_sharpness_ranks_blur_below_detail():
    sharp = noise_frame(0)
    assert sharpness(sharp) > sharpness(blurred(sharp))


def test_sharpness_accepts_grayscale():
    gray = cv2.cvtColor(noise_frame(0), cv2.COLOR_BGR2GRAY)
    assert sharpness(gray) > 0


# --- selection ---------------------------------------------------------------


def test_short_input_is_returned_whole():
    frames = as_candidates([noise_frame(i) for i in range(3)])
    assert select_frames(frames, n_frames=10, min_frames=4) == [0, 1, 2]


def test_selection_respects_the_frame_budget():
    frames = as_candidates([noise_frame(i) for i in range(60)])
    assert len(select_frames(frames, n_frames=10)) <= 10


def test_blurred_frames_lose_to_sharp_ones_in_the_same_window():
    """Every other frame is blurred; the selection should prefer the sharp ones."""
    frames = []
    for i in range(24):
        f = noise_frame(i)
        frames.append(blurred(f) if i % 2 == 0 else f)

    chosen = select_frames(as_candidates(frames), n_frames=12, dedup_correlation=1.1)
    odd = sum(1 for i in chosen if i % 2 == 1)
    assert odd > len(chosen) / 2, "selection favoured blurred frames"


def test_near_duplicates_are_skipped():
    """A long static stretch should not consume the whole budget."""
    still = noise_frame(0)
    frames = [still.copy() for _ in range(20)] + [noise_frame(i) for i in range(1, 21)]

    chosen = select_frames(as_candidates(frames), n_frames=20, dedup_correlation=0.98)
    from_static = sum(1 for i in chosen if i < 20)
    assert from_static <= 2, f"kept {from_static} near-identical frames"


def test_dedup_never_starves_below_min_frames():
    frames = [noise_frame(0).copy() for _ in range(30)]      # all identical
    chosen = select_frames(as_candidates(frames), n_frames=10, min_frames=4)
    assert len(chosen) >= 4


def test_selection_is_sorted_and_unique():
    frames = as_candidates([noise_frame(i) for i in range(40)])
    chosen = select_frames(frames, n_frames=12)
    assert chosen == sorted(set(chosen))


def test_blur_floor_never_empties_a_window():
    """If a whole window is blurry, take its best frame rather than nothing."""
    frames = [blurred(noise_frame(i)) for i in range(20)]
    chosen = select_frames(as_candidates(frames), n_frames=8, dedup_correlation=1.1)
    assert len(chosen) >= 4


# --- extract_frames ----------------------------------------------------------


def test_extract_from_image_folder_writes_pngs(tmp_path):
    src = tmp_path / "images"
    src.mkdir()
    for i in range(8):
        cv2.imwrite(str(src / f"{i:02d}.jpg"), noise_frame(i))

    out = tmp_path / "frames"
    extracted = extract_frames(src, out, n_frames=4, dedup_correlation=1.1)

    assert 1 <= len(extracted) <= 4
    for f in extracted:
        assert f.path.exists() and f.path.suffix == ".png"
        assert cv2.imread(str(f.path)) is not None
    assert [f.index for f in extracted] == list(range(len(extracted)))


def test_extract_rejects_unsupported_source(tmp_path):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("hello")
    with pytest.raises(ValueError):
        extract_frames(bogus, tmp_path / "out")


def test_extract_rejects_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError):
        extract_frames(empty, tmp_path / "out")


# --- rotation ----------------------------------------------------------------


def test_apply_rotation_90_transposes_dimensions():
    frame = np.zeros((100, 50, 3), dtype=np.uint8)     # tall
    assert _apply_rotation(frame, 90).shape[:2] == (50, 100)
    assert _apply_rotation(frame, 270).shape[:2] == (50, 100)


def test_apply_rotation_180_preserves_dimensions():
    frame = np.zeros((100, 50, 3), dtype=np.uint8)
    assert _apply_rotation(frame, 180).shape[:2] == (100, 50)


def test_apply_rotation_zero_is_identity():
    frame = noise_frame(0)
    assert np.array_equal(_apply_rotation(frame, 0), frame)


def test_apply_rotation_ignores_unknown_angles():
    frame = noise_frame(0)
    assert np.array_equal(_apply_rotation(frame, 45), frame)


def test_rotation_directions_are_inverses():
    frame = noise_frame(0)
    assert np.array_equal(_apply_rotation(_apply_rotation(frame, 90), 270), frame)


def test_apply_rotation_90_actually_moves_content():
    """Dimension checks alone pass for a transpose *or* a real rotation; this
    pins the corner that must end up top-left."""
    frame = np.zeros((4, 2, 3), dtype=np.uint8)
    frame[3, 0] = 255                                   # bottom-left
    rotated = _apply_rotation(frame, 90)                # clockwise
    assert tuple(rotated[0, 0]) == (255, 255, 255)      # -> top-left
