"""Guards that must fire *before* half an hour of inference is spent.

The mixed-shape check exists because it originally did not: a folder of mixed
portrait/landscape photos ran 30 forward passes and then died stacking arrays.
"""

import numpy as np
import pytest

from room3d.reconstruct import _enforce_uniform_shape, _rescale


def fake_image(h, w, idx=0):
    """Mirrors what dust3r's load_images returns, including idx/instance."""
    return {
        "img": np.zeros((1, 3, h, w), dtype=np.float32),
        "true_shape": np.array([[h, w]], dtype=np.int32),
        "idx": idx,
        "instance": str(idx),
    }


def fake_images(*shapes):
    return [fake_image(h, w, idx=i) for i, (h, w) in enumerate(shapes)]


def test_uniform_input_passes_through_untouched():
    images = fake_images(*[(512, 384)] * 4)
    kept, source_indices = _enforce_uniform_shape(images, allow_mixed=False, verbose=False)
    assert kept is images
    assert source_indices == [0, 1, 2, 3]


def test_mixed_shapes_raise_by_default():
    images = fake_images((512, 384), (384, 512), (512, 384))
    with pytest.raises(ValueError, match="mixed shapes"):
        _enforce_uniform_shape(images, allow_mixed=False, verbose=False)


def test_mixed_shape_error_names_the_escape_hatch():
    images = fake_images((512, 384), (384, 512))
    with pytest.raises(ValueError, match=r"--allow-mixed"):
        _enforce_uniform_shape(images, allow_mixed=False, verbose=False)


def test_allow_mixed_keeps_the_majority_orientation():
    images = fake_images((512, 384), (384, 512), (512, 384))
    kept, _ = _enforce_uniform_shape(images, allow_mixed=True, verbose=False)
    assert len(kept) == 2
    assert all(im["img"].shape[-2:] == (512, 384) for im in kept)


def test_allow_mixed_renumbers_survivors_contiguously():
    """dust3r's global aligner asserts edge indices are a contiguous range, so
    dropping image 1 must renumber 2->1 rather than leave a gap."""
    images = fake_images((512, 384), (384, 512), (512, 384), (512, 384))
    kept, _ = _enforce_uniform_shape(images, allow_mixed=True, verbose=False)

    assert [im["idx"] for im in kept] == [0, 1, 2]
    assert [im["instance"] for im in kept] == ["0", "1", "2"]


def test_allow_mixed_reports_original_indices_for_provenance():
    """Renumbering makes idx contiguous, so the mapping back to the frame files
    on disk has to be returned separately or it is lost."""
    images = fake_images((512, 384), (384, 512), (512, 384), (512, 384))
    _, source_indices = _enforce_uniform_shape(images, allow_mixed=True, verbose=False)
    assert source_indices == [0, 2, 3]


def test_allow_mixed_preserves_input_order():
    images = fake_images((512, 384), (384, 512), (512, 384))
    kept, _ = _enforce_uniform_shape(images, allow_mixed=True, verbose=False)
    assert np.array_equal(kept[0]["img"], images[0]["img"])
    assert np.array_equal(kept[1]["img"], images[2]["img"])


def test_allow_mixed_does_not_mutate_the_input():
    images = fake_images((512, 384), (384, 512), (512, 384))
    _enforce_uniform_shape(images, allow_mixed=True, verbose=False)
    assert [im["idx"] for im in images] == [0, 1, 2], "input list was mutated"


def test_mixed_shape_error_names_the_offending_files():
    images = fake_images((512, 384), (384, 512), (512, 384))
    paths = ["a.jpg", "sideways.jpg", "c.jpg"]
    with pytest.raises(ValueError, match="sideways.jpg"):
        _enforce_uniform_shape(images, paths, allow_mixed=False, verbose=False)


def test_allow_mixed_still_fails_when_too_few_survive():
    images = fake_images((512, 384), (384, 512))
    with pytest.raises(ValueError, match="need at least 2"):
        _enforce_uniform_shape(images, allow_mixed=True, verbose=False)


# --- scale reference ---------------------------------------------------------


def test_rescale_sets_the_largest_extent():
    pts = np.zeros((1, 2, 2, 3), dtype=np.float32)
    pts[0, 0, 0] = [0, 0, 0]
    pts[0, 1, 1] = [2, 1, 1]           # largest extent is 2 along x
    poses = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1))
    poses[0, :3, 3] = [1, 0, 0]

    scaled_pts, scaled_poses, factor = _rescale(pts, poses, target_largest_extent=4.0)

    assert factor == pytest.approx(2.0)
    finite = scaled_pts[np.isfinite(scaled_pts).all(axis=-1)]
    assert (finite.max(axis=0) - finite.min(axis=0)).max() == pytest.approx(4.0)
    assert np.allclose(scaled_poses[0, :3, 3], [2, 0, 0]), "poses must scale with the cloud"


def test_rescale_is_a_noop_on_a_degenerate_cloud():
    pts = np.zeros((1, 2, 2, 3), dtype=np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1))
    _, _, factor = _rescale(pts, poses, 4.0)
    assert factor == 1.0


# --- environment resolution --------------------------------------------------


def test_resolve_api_key_prefers_the_canonical_name(monkeypatch):
    from room3d.vlm import resolve_api_key

    monkeypatch.setenv("GEMINI_API_KEY", "canonical")
    monkeypatch.setenv("API_KEY", "generic")
    assert resolve_api_key() == "canonical"


def test_resolve_api_key_accepts_generic_name(monkeypatch):
    from room3d.vlm import resolve_api_key

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "generic")
    assert resolve_api_key() == "generic"


def test_resolve_api_key_ignores_blank_values(monkeypatch):
    from room3d.vlm import resolve_api_key

    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("API_KEY", "real")
    assert resolve_api_key() == "real"


def test_resolve_api_key_returns_none_when_unset(monkeypatch):
    from room3d.vlm import resolve_api_key

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert resolve_api_key() is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("models/gemini-2.5-flash", "gemini-2.5-flash"),
        ("google/gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
        ("gemini/gemini-2.5-flash", "gemini-2.5-flash"),
        ("  gemini-3.6-flash  ", "gemini-3.6-flash"),
        ("gemini-2.5-flash", "gemini-2.5-flash"),
    ],
)
def test_strip_model_prefix(raw, expected):
    """LiteLLM-style .env values carry a routing prefix the google-genai SDK
    rejects with a bare 404, which is a miserable error to diagnose."""
    from room3d.vlm import strip_model_prefix

    assert strip_model_prefix(raw) == expected


def test_strip_model_prefix_leaves_the_model_name_intact():
    """'gemini-...' must not be eaten by the 'gemini/' prefix rule."""
    from room3d.vlm import strip_model_prefix

    assert strip_model_prefix("gemini-3.5-flash-lite") == "gemini-3.5-flash-lite"


def test_resolve_model_strips_prefix_from_env(monkeypatch):
    from room3d.vlm import resolve_model

    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("MODEL", "google/gemini-3.5-flash-lite")
    assert resolve_model() == "gemini-3.5-flash-lite"


def test_resolve_model_falls_back_to_default(monkeypatch):
    from room3d.vlm import resolve_model

    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    assert resolve_model("fallback") == "fallback"

