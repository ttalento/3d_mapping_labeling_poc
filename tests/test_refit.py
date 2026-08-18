"""Re-fitting a labelled room must cost nothing and change only the geometry.

Detections are the expensive part of a run -- one VLM call per frame -- and they
are already on disk in `observations.json` as pixel boxes. Everything downstream
of them is arithmetic, so improving the box fit must not mean paying for
detection again.
"""

import json

import numpy as np
import pytest

from room3d.artifacts import load_observation_points, save_frames_npz
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.session import LabelingSession
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from room3d.level import UP_VECTOR
from room3d.projection import points_inside_fraction
from room3d.refit import refit_room
from test_boxfit_wiring import build_recon, labelled_session, sofa_detection


def legacy_room(tmp_path, mangle_boxes=True):
    """A room labelled the old way: observations on disk, no points sidecar."""
    session = labelled_session()
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)

    save_frames_npz(tmp_path / "frames.npz", session.recon)
    (tmp_path / "observations.json").write_text(
        json.dumps(build_observations_doc(session, "room"))
    )
    (tmp_path / "objects.json").write_text(
        json.dumps({"room": "room", "units": "meters", "objects": []})
    )
    return tmp_path


def test_refit_rebuilds_objects_without_any_detector(tmp_path):
    room = legacy_room(tmp_path)
    result = refit_room(room, verbose=False)

    assert result["n_observations"] == 4
    written = json.loads((room / "objects.json").read_text())
    assert len(written["objects"]) == 1


def test_refit_produces_level_boxes(tmp_path):
    room = legacy_room(tmp_path)
    refit_room(room, verbose=False)

    obb = json.loads((room / "objects.json").read_text())["objects"][0]["obb"]
    assert np.allclose(np.asarray(obb["R"])[:, 1], UP_VECTOR, atol=1e-2)


def test_refit_writes_the_points_sidecar_so_the_viewer_can_refuse(tmp_path):
    room = legacy_room(tmp_path)
    refit_room(room, verbose=False)

    points = load_observation_points(room / "observations.json")
    assert len(points) == 4
    assert all(p.shape[1] == 3 for p in points.values())


def test_refit_boxes_contain_the_geometry_they_describe(tmp_path):
    room = legacy_room(tmp_path)
    refit_room(room, verbose=False)

    obj = json.loads((room / "objects.json").read_text())["objects"][0]
    pooled = np.vstack(list(load_observation_points(room / "observations.json").values()))

    from room3d.projection import OrientedBox

    obb = OrientedBox(
        np.asarray(obj["obb"]["center"]),
        np.asarray(obj["obb"]["extent"]),
        np.asarray(obj["obb"]["R"]),
    )
    assert points_inside_fraction(obb, pooled) > 0.9


def test_refit_keeps_the_labels_the_vlm_gave(tmp_path):
    room = legacy_room(tmp_path)
    refit_room(room, verbose=False)

    written = json.loads((room / "objects.json").read_text())
    assert written["objects"][0]["label"] == "sofa"


def test_refit_preserves_run_metadata_from_the_previous_objects_file(tmp_path):
    room = legacy_room(tmp_path)
    (room / "objects.json").write_text(
        json.dumps({"room": "living", "units": "meters", "scale_verified": True,
                    "n_frames_labeled": 4, "objects": []})
    )
    refit_room(room, verbose=False)

    written = json.loads((room / "objects.json").read_text())
    assert written["scale_verified"] is True
    assert written["room"] == "living"
    assert written["n_frames_labeled"] == 4


def test_refit_backs_up_both_files_it_overwrites(tmp_path):
    """Detections are the irreplaceable part of a run. Rewriting the file that
    holds them without a copy would make a bad refit unrecoverable."""
    room = legacy_room(tmp_path)
    before = json.loads((room / "observations.json").read_text())

    refit_room(room, verbose=False)

    assert (room / "objects.prev.json").exists()
    backup = json.loads((room / "observations.prev.json").read_text())
    assert backup == before


def test_refit_refuses_a_room_with_no_observations(tmp_path):
    save_frames_npz(tmp_path / "frames.npz", build_recon())
    with pytest.raises(FileNotFoundError, match="observations.json"):
        refit_room(tmp_path, verbose=False)


def test_refit_refuses_a_room_with_no_reconstruction(tmp_path):
    (tmp_path / "observations.json").write_text(json.dumps({"observations": []}))
    with pytest.raises(FileNotFoundError, match="frames.npz"):
        refit_room(tmp_path, verbose=False)


def test_refit_skips_observations_whose_pixel_box_was_never_recorded(tmp_path):
    room = legacy_room(tmp_path)
    doc = json.loads((room / "observations.json").read_text())
    doc["observations"][0]["box_px"] = None
    (room / "observations.json").write_text(json.dumps(doc))

    result = refit_room(room, verbose=False)
    assert result["n_observations"] == 3
    assert result["n_skipped"] == 1
