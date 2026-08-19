"""Gate 8e: promoting a query result into the room.

This is the step that fixes duplicates. A query that finds one couch where the
object list holds two must be able to say so destructively -- with a backup,
because a vague phrase must never cost you a labelled room.
"""

import json

import numpy as np
import pytest

from room3d.artifacts import save_frames_npz
from room3d.consensus import View
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from room3d.projection import OrientedBox
from room3d.query import QueryMatch, commit_match, query_room
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


# --- absorbed observations must not dangle ------------------------------------


def two_object_room(tmp_path):
    """A hand-built room with two objects, each seen twice, so committing a
    match that absorbs both leaves observations pointing at both old ids."""
    obs = [
        {"id": 0, "object_id": "obj_000", "frame_idx": 0, "label": "couch",
         "vlm_confidence": 0.9, "box_px": [1, 2, 3, 4]},
        {"id": 1, "object_id": "obj_000", "frame_idx": 1, "label": "couch",
         "vlm_confidence": 0.9, "box_px": [1, 2, 3, 4]},
        {"id": 2, "object_id": "obj_001", "frame_idx": 2, "label": "couch",
         "vlm_confidence": 0.9, "box_px": [1, 2, 3, 4]},
        {"id": 3, "object_id": "obj_001", "frame_idx": 3, "label": "couch",
         "vlm_confidence": 0.9, "box_px": [1, 2, 3, 4]},
    ]
    (tmp_path / "observations.json").write_text(json.dumps({
        "room": "room", "image_hw": [10, 10], "frames_labeled": [0, 1, 2, 3],
        "observations": obs,
    }))

    def obj(object_id):
        return {
            "id": object_id, "label": "couch", "aliases": [],
            "centroid": [0.0, 0.0, 0.0],
            "obb": {"center": [0.0, 0.0, 0.0], "extent": [1.0, 1.0, 1.0],
                    "R": np.eye(3).tolist()},
            "confidence": 0.5, "n_observations": 2, "seen_in": [0, 1],
            "observation_ids": [0, 1] if object_id == "obj_000" else [2, 3],
        }

    (tmp_path / "objects.json").write_text(json.dumps({
        "room": "room", "units": "meters", "scale_verified": False,
        "n_frames_labeled": 4, "objects": [obj("obj_000"), obj("obj_001")],
    }))
    return tmp_path


def absorbing_match():
    obb = OrientedBox(np.zeros(3), np.ones(3), np.eye(3))
    return QueryMatch(
        label="couch",
        obb=obb,
        score=0.8,
        views=[
            View(0, (1, 2, 3, 4), label="couch", observation_id=0, object_id="obj_000"),
            View(1, (1, 2, 3, 4), label="couch", observation_id=1, object_id="obj_000"),
            View(2, (1, 2, 3, 4), label="couch", observation_id=2, object_id="obj_001"),
            View(3, (1, 2, 3, 4), label="couch", observation_id=3, object_id="obj_001"),
        ],
        n_points=100,
        vote_stats={"mean_vote": 0.9, "min_vote": 0.5,
                    "n_candidates": 100, "n_kept": 100, "kept_frac": 1.0},
        absorbed_object_ids=["obj_000", "obj_001"],
        supported=True,
    )


def test_committing_a_multi_object_match_repoints_every_absorbed_observation(tmp_path):
    room = two_object_room(tmp_path)
    result = commit_match(room, absorbing_match(), verbose=False)

    observations = json.loads((room / "observations.json").read_text())["observations"]
    live_ids = {o["id"] for o in json.loads((room / "objects.json").read_text())["objects"]}
    # An id can be reused (the committed object may take an absorbed id), so
    # the real invariant is that every observation names an id that still
    # exists in objects.json -- not merely that it avoids the removed set.
    assert all(o["object_id"] in live_ids for o in observations)
    assert all(o["object_id"] == result["object_id"] for o in observations)


def test_committing_backs_up_observations_json_before_repointing(tmp_path):
    room = two_object_room(tmp_path)
    before = json.loads((room / "observations.json").read_text())

    commit_match(room, absorbing_match(), verbose=False)

    assert json.loads((room / "observations.prev.json").read_text()) == before
