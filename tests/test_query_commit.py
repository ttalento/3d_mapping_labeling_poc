"""Gate 8e: promoting a query result into the room.

This is the step that fixes duplicates. A query that finds one couch where the
object list holds two must be able to say so destructively -- with a backup,
because a vague phrase must never cost you a labelled room.
"""

import json

import numpy as np
import pytest

from room3d.artifacts import save_frames_npz
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from room3d.query import commit_match, query_room
from test_boxfit_wiring import labelled_session


def room_on_disk(tmp_path, extra_objects=()):
    session = labelled_session()
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)

    save_frames_npz(tmp_path / "frames.npz", session.recon)
    (tmp_path / "observations.json").write_text(
        json.dumps(build_observations_doc(session, "room"))
    )
    objects = [o.as_dict() for o in session.objects] + list(extra_objects)
    (tmp_path / "objects.json").write_text(
        json.dumps({
            "room": "room", "units": "meters", "scale_verified": True,
            "n_frames_labeled": 4, "objects": objects,
        })
    )
    return tmp_path


def decoy(object_id="obj_099"):
    return {
        "id": object_id, "label": "lamp", "aliases": [],
        "centroid": [9.0, 9.0, 9.0],
        "obb": {"center": [9.0, 9.0, 9.0], "extent": [1.0, 1.0, 1.0],
                "R": np.eye(3).tolist()},
        "confidence": 0.5, "n_observations": 1, "seen_in": [0],
        "observation_ids": [],
    }


def test_committing_writes_the_match_into_objects_json(tmp_path):
    room = room_on_disk(tmp_path)
    match = query_room(room, "sofa", verbose=False).matches[0]
    commit_match(room, match, verbose=False)

    written = json.loads((room / "objects.json").read_text())["objects"]
    entry = next(o for o in written if o["label"] == "sofa")
    assert np.allclose(entry["obb"]["extent"], match.obb.extent)


def test_committing_removes_the_objects_the_match_absorbed(tmp_path):
    room = room_on_disk(tmp_path, extra_objects=[decoy()])
    before = len(json.loads((room / "objects.json").read_text())["objects"])

    match = query_room(room, "sofa", verbose=False).matches[0]
    assert match.absorbed_object_ids == ["obj_000"]
    result = commit_match(room, match, verbose=False)

    after = json.loads((room / "objects.json").read_text())["objects"]
    assert len(after) == before                      # one removed, one added
    assert result["removed"] == ["obj_000"]


def test_committing_leaves_unrelated_objects_alone(tmp_path):
    room = room_on_disk(tmp_path, extra_objects=[decoy()])
    match = query_room(room, "sofa", verbose=False).matches[0]
    commit_match(room, match, verbose=False)

    written = json.loads((room / "objects.json").read_text())["objects"]
    assert any(o["id"] == "obj_099" for o in written)


def test_committing_backs_up_the_previous_objects_file(tmp_path):
    room = room_on_disk(tmp_path)
    before = json.loads((room / "objects.json").read_text())

    match = query_room(room, "sofa", verbose=False).matches[0]
    commit_match(room, match, verbose=False)

    assert json.loads((room / "objects.prev.json").read_text()) == before


def test_committing_preserves_run_metadata(tmp_path):
    room = room_on_disk(tmp_path)
    match = query_room(room, "sofa", verbose=False).matches[0]
    commit_match(room, match, verbose=False)

    written = json.loads((room / "objects.json").read_text())
    assert written["scale_verified"] is True
    assert written["room"] == "room"


def test_the_committed_object_reuses_an_absorbed_id_so_ids_stay_stable(tmp_path):
    room = room_on_disk(tmp_path)
    match = query_room(room, "sofa", verbose=False).matches[0]
    assert commit_match(room, match, verbose=False)["object_id"] == "obj_000"


def test_a_match_absorbing_nothing_gets_a_fresh_unused_id(tmp_path):
    room = room_on_disk(tmp_path, extra_objects=[decoy("obj_005")])
    match = query_room(room, "sofa", verbose=False).matches[0]
    match.absorbed_object_ids = []

    new_id = commit_match(room, match, verbose=False)["object_id"]
    existing = {o["id"] for o in json.loads((room / "objects.json").read_text())["objects"]}
    assert new_id in existing
    assert new_id not in {"obj_000", "obj_005"}


def test_an_unsupported_match_cannot_be_committed(tmp_path):
    room = room_on_disk(tmp_path)
    result = query_room(room, "sofa", config_overrides={"min_vote": 1.01}, verbose=False)
    with pytest.raises(ValueError, match="unsupported"):
        commit_match(room, result.matches[0], verbose=False)
