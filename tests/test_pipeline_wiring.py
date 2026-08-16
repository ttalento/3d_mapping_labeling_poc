"""End-to-end wiring of the labeling half, with the VLM and LLM stubbed out.

This is the test that catches integration bugs -- a transposed box, a dropped
argument, a mask decoded against the wrong image size -- without spending API
quota or waiting on a reconstruction. It runs the real tools, the real
projection and the real fusion over a synthetic room whose answer is known.
"""

import base64
import io
import json

import numpy as np
import pytest

from room3d.artifacts import (
    Reconstruction,
    load_frames_npz,
    save_frames_npz,
    save_ply,
    save_trajectory_tum,
)
from room3d.config import LabelConfig
from room3d.crew.semantics import SynonymResolver, _parse_groups, build_synonym_resolver
from room3d.crew.session import LabelingSession, select_covering_frames
from room3d.vlm import Detection

H, W, N = 48, 64, 4

# Two objects in a synthetic room, at known world positions.
DESK_XYZ = np.array([0.0, 0.0, 2.0])
LAMP_XYZ = np.array([2.0, 0.0, 2.0])
BACKGROUND_Z = 6.0


def build_recon() -> Reconstruction:
    """Four cameras spread along x, each seeing both objects at fixed world points."""
    pts3d = np.zeros((N, H, W, 3), dtype=np.float32)
    pts3d[..., 2] = BACKGROUND_Z

    # Left half of every frame is the desk; right half is the lamp.
    pts3d[:, 12:36, 8:28] = DESK_XYZ
    pts3d[:, 12:36, 36:56] = LAMP_XYZ

    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, 0, 3] = np.linspace(-1.0, 1.0, N)      # cameras spread along x

    intrinsics = np.tile(
        np.array([[50, 0, W / 2], [0, 50, H / 2], [0, 0, 1]], dtype=np.float32), (N, 1, 1)
    )

    return Reconstruction(
        images=np.full((N, H, W, 3), 128, dtype=np.uint8),
        pts3d=pts3d,
        conf_mask=np.ones((N, H, W), dtype=bool),
        poses=poses,
        intrinsics=intrinsics,
        frame_ids=np.arange(N, dtype=np.int32),
    )


def px_to_gemini_box(x0, y0, x1, y1):
    """Pixel box -> Gemini's [ymin, xmin, ymax, xmax] at 0-1000."""
    return (
        int(round(y0 / H * 1000)), int(round(x0 / W * 1000)),
        int(round(y1 / H * 1000)), int(round(x1 / W * 1000)),
    )


def png_mask_b64(box_px, filled=True):
    from PIL import Image

    x0, y0, x1, y1 = box_px
    arr = np.full((y1 - y0, x1 - x0), 255 if filled else 0, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class StubDetector:
    """Stands in for GeminiDetector, returning the same shapes it would."""

    def __init__(self, labels=("desk", "lamp"), with_masks=True):
        self.labels = labels
        self.with_masks = with_masks
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        desk_box = px_to_gemini_box(8, 12, 28, 36)
        lamp_box = px_to_gemini_box(36, 12, 56, 36)

        out = []
        for label, box in ((self.labels[0], desk_box), (self.labels[1], lamp_box)):
            out.append(
                Detection(
                    label=label,
                    box_2d=box,
                    confidence=0.9,
                    mask_b64=png_mask_b64(
                        (
                            int(box[1] / 1000 * W), int(box[0] / 1000 * H),
                            int(np.ceil(box[3] / 1000 * W)), int(np.ceil(box[2] / 1000 * H)),
                        )
                    ) if self.with_masks else None,
                )
            )
        return out


def run_direct(session, detector, llm=None):
    from room3d.crew.tools import (
        ClusterObservationsTool,
        DetectObjectsTool,
        ProjectDetectionsTool,
        SelectFramesTool,
    )

    SelectFramesTool(session)._run(session.config.max_frames)
    DetectObjectsTool(session, detector)._run()
    ProjectDetectionsTool(session)._run()
    return ClusterObservationsTool(session, llm)._run(use_llm_synonyms=llm is not None)


@pytest.fixture
def session():
    return LabelingSession(
        recon=build_recon(),
        config=LabelConfig(max_frames=N, min_mask_points=20, mask_erode_px=1),
    )


# --- the headline wiring test ------------------------------------------------


def test_full_labeling_produces_two_objects_at_the_right_places(session):
    raw = run_direct(session, StubDetector())
    objects = json.loads(raw)

    assert len(objects) == 2, f"expected 2 objects, got {[o['label'] for o in objects]}"

    by_label = {o["label"]: o for o in objects}
    assert set(by_label) == {"desk", "lamp"}

    assert np.allclose(by_label["desk"]["centroid"], DESK_XYZ, atol=0.05)
    assert np.allclose(by_label["lamp"]["centroid"], LAMP_XYZ, atol=0.05)

    for obj in objects:
        assert obj["n_observations"] == N
        assert obj["seen_in"] == list(range(N))
        assert 0.0 <= obj["confidence"] <= 1.0


def test_detections_are_made_once_per_selected_frame(session):
    detector = StubDetector()
    run_direct(session, detector)
    assert detector.calls == N


def test_box_fallback_matches_mask_result_on_a_flat_object(session):
    """With no mask returned, the box path must still find the object.

    The synthetic objects are flat and fill their boxes exactly, so the two
    paths should agree here; a disagreement means the box fallback is wired
    to different pixels than the mask path.
    """
    with_mask = json.loads(run_direct(session, StubDetector(with_masks=True)))

    fresh = LabelingSession(recon=build_recon(), config=session.config)
    without_mask = json.loads(run_direct(fresh, StubDetector(with_masks=False)))

    a = {o["label"]: np.asarray(o["centroid"]) for o in with_mask}
    b = {o["label"]: np.asarray(o["centroid"]) for o in without_mask}
    assert set(a) == set(b)
    for label in a:
        assert np.allclose(a[label], b[label], atol=0.05)


def test_synonyms_across_frames_collapse_to_one_object(session):
    """Frames disagreeing on the name must not produce two objects."""
    class Alternating(StubDetector):
        def detect(self, image):
            names = ("desk", "lamp") if self.calls % 2 == 0 else ("table", "lamp")
            self.labels = names
            return super().detect(image)

    objects = json.loads(run_direct(session, Alternating()))
    assert len(objects) == 2
    desk = next(o for o in objects if o["label"] in {"desk", "table"})
    assert desk["n_observations"] == N
    assert "table" in desk["aliases"] or "desk" in desk["aliases"]


def test_zero_detections_yields_no_objects(session):
    class Empty(StubDetector):
        def detect(self, image):
            return []

    assert json.loads(run_direct(session, Empty())) == []


def test_unsupported_geometry_is_dropped_not_fabricated(session):
    session.recon.conf_mask[:] = False
    assert json.loads(run_direct(session, StubDetector())) == []
    assert session.objects == []


# --- frame selection ---------------------------------------------------------


def test_select_covering_frames_returns_all_when_budget_exceeds_supply():
    recon = build_recon()
    assert select_covering_frames(recon, 99) == list(range(N))


def test_select_covering_frames_spreads_over_the_camera_path():
    """Sampling is seeded at the most central camera and then walks outward, so
    the guarantee is wide spread, not specific indices."""
    recon = build_recon()
    centres = recon.poses[:, 0, 3]
    full_span = centres.max() - centres.min()

    chosen = select_covering_frames(recon, 2)
    assert len(chosen) == 2

    covered = centres[chosen].max() - centres[chosen].min()
    assert covered >= 0.6 * full_span, "sampler picked two nearby frames"


def test_select_covering_frames_is_sorted_and_unique():
    recon = build_recon()
    chosen = select_covering_frames(recon, 3)
    assert chosen == sorted(set(chosen))


def test_select_covering_frames_honours_budget_with_coincident_cameras():
    """A person standing still produces near-identical camera centres, which
    stalls farthest-point sampling. The budget must still be filled."""
    recon = build_recon()
    recon.poses[:, :3, 3] = 0.0            # every camera at the same place

    for budget in (1, 2, 3, N):
        chosen = select_covering_frames(recon, budget)
        assert len(chosen) == budget, f"budget {budget} gave {chosen}"
        assert chosen == sorted(set(chosen))


# --- artifacts round trip ----------------------------------------------------


def test_frames_npz_round_trip(tmp_path):
    original = build_recon()
    path = tmp_path / "frames.npz"
    save_frames_npz(path, original)
    loaded = load_frames_npz(path)

    assert np.array_equal(loaded.images, original.images)
    assert np.allclose(loaded.pts3d, original.pts3d)
    assert np.array_equal(loaded.conf_mask, original.conf_mask)
    assert np.allclose(loaded.poses, original.poses)
    assert loaded.n_frames == N and loaded.image_hw == (H, W)


def test_reconstruction_rejects_inconsistent_shapes():
    with pytest.raises(ValueError):
        Reconstruction(
            images=np.zeros((N, H, W, 3), np.uint8),
            pts3d=np.zeros((N, H, W, 3), np.float32),
            conf_mask=np.zeros((N, H, W), bool),
            poses=np.zeros((N - 1, 4, 4), np.float32),      # wrong N
            intrinsics=np.zeros((N, 3, 3), np.float32),
            frame_ids=np.arange(N, dtype=np.int32),
        )


def test_save_ply_is_readable_and_preserves_count(tmp_path):
    pts = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    cols = np.random.default_rng(1).integers(0, 256, (500, 3), dtype=np.uint8)
    path = tmp_path / "cloud.ply"
    save_ply(path, pts, cols)

    header = path.read_bytes()[:200].decode("ascii", errors="ignore")
    assert "element vertex 500" in header
    assert "binary_little_endian" in header


def test_save_ply_rejects_mismatched_colors(tmp_path):
    with pytest.raises(ValueError):
        save_ply(tmp_path / "x.ply", np.zeros((10, 3)), np.zeros((9, 3), np.uint8))


def test_trajectory_is_valid_tum_with_unit_quaternions(tmp_path):
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[1, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])   # 90 deg about z
    poses[2, :3, 3] = [1.0, 2.0, 3.0]

    path = tmp_path / "traj.txt"
    save_trajectory_tum(path, poses)
    lines = path.read_text().strip().splitlines()

    assert len(lines) == 3
    for line in lines:
        parts = line.split()
        assert len(parts) == 8, "TUM format is: t x y z qx qy qz qw"
        q = np.array([float(v) for v in parts[4:]])
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6

    assert [float(v) for v in lines[2].split()[1:4]] == [1.0, 2.0, 3.0]


# --- synonym resolver --------------------------------------------------------


def test_synonym_resolver_groups_and_canonicalises():
    r = SynonymResolver([["monitor", "display", "screen"]])
    assert r.compatible("monitor", "display")
    assert r.compatible("display", "screen")
    assert not r.compatible("monitor", "chair")
    assert r.canonicalize(["display", "screen"]) == "monitor"


def test_synonym_resolver_falls_back_for_unknown_labels():
    r = SynonymResolver([["monitor", "display"]])
    assert r.compatible("sofa", "couch")        # static table still applies
    assert not r.compatible("sofa", "monitor")


def test_build_synonym_resolver_without_llm_uses_static_table():
    r = build_synonym_resolver(["sofa", "couch", "desk"], None)
    assert r.compatible("sofa", "couch")


def test_build_synonym_resolver_survives_a_failing_llm():
    def broken(_prompt):
        raise RuntimeError("429 rate limited")

    r = build_synonym_resolver(["monitor", "display"], broken)
    assert r.compatible("monitor", "display")   # degraded to the static table


def test_build_synonym_resolver_uses_llm_grouping():
    def fake(_prompt):
        return json.dumps(
            {"groups": [{"canonical": "wombat", "labels": ["wombat", "quokka"]}]}
        )

    r = build_synonym_resolver(["wombat", "quokka"], fake)
    assert r.compatible("wombat", "quokka")


def test_parse_groups_strips_markdown_fences():
    raw = '```json\n{"groups": [{"canonical": "desk", "labels": ["desk", "table"]}]}\n```'
    groups = _parse_groups(raw, ["desk", "table"])
    assert groups == {"desk": ["table"]}


def test_parse_groups_keeps_labels_the_model_omitted():
    raw = '{"groups": [{"canonical": "desk", "labels": ["desk"]}]}'
    groups = _parse_groups(raw, ["desk", "chair"])
    assert set(groups) == {"desk", "chair"}


def test_parse_groups_ignores_hallucinated_labels():
    raw = '{"groups": [{"canonical": "desk", "labels": ["desk", "spaceship"]}]}'
    groups = _parse_groups(raw, ["desk"])
    assert groups == {"desk": []}


def test_parse_groups_assigns_each_label_once():
    raw = (
        '{"groups": [{"canonical": "desk", "labels": ["desk", "table"]},'
        ' {"canonical": "table", "labels": ["table"]}]}'
    )
    groups = _parse_groups(raw, ["desk", "table"])
    assigned = [l for members in groups.values() for l in members] + list(groups)
    assert sorted(assigned) == ["desk", "table"]
