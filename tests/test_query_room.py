"""Gate 8c: the whole query, end to end, on a room on disk.

Reuses the synthetic levelled room from the box-fit tests so the query is
exercised against the same geometry the pipeline produces.
"""

import json

import numpy as np
import pytest

from room3d.artifacts import save_frames_npz
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from room3d.level import UP_VECTOR
from room3d.query import QueryMatch, query_room
from test_boxfit_wiring import labelled_session


def room_on_disk(tmp_path):
    session = labelled_session()
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)

    save_frames_npz(tmp_path / "frames.npz", session.recon)
    (tmp_path / "observations.json").write_text(
        json.dumps(build_observations_doc(session, "room"))
    )
    (tmp_path / "objects.json").write_text(
        json.dumps({
            "room": "room", "units": "meters", "scale_verified": False,
            "n_frames_labeled": 4,
            "objects": [o.as_dict() for o in session.objects],
        })
    )
    return tmp_path


def test_querying_a_labelled_object_returns_one_match(tmp_path):
    result = query_room(room_on_disk(tmp_path), "sofa", verbose=False)
    assert result.source == "cache"
    assert len(result.matches) == 1
    assert result.matches[0].label == "sofa"


def test_the_match_carries_a_levelled_box(tmp_path):
    match = query_room(room_on_disk(tmp_path), "sofa", verbose=False).matches[0]
    assert match.obb is not None
    assert np.allclose(match.obb.R[:, 1], UP_VECTOR, atol=1e-2)


def test_a_synonym_finds_the_same_object(tmp_path):
    room = room_on_disk(tmp_path)
    a = query_room(room, "sofa", verbose=False).matches[0]
    b = query_room(room, "couch", verbose=False).matches[0]
    assert np.allclose(a.obb.extent, b.obb.extent)


def test_the_match_names_the_objects_it_would_replace(tmp_path):
    match = query_room(room_on_disk(tmp_path), "sofa", verbose=False).matches[0]
    assert match.absorbed_object_ids == ["obj_000"]


def test_an_unknown_phrase_with_no_detector_says_so_rather_than_saying_absent(tmp_path):
    result = query_room(room_on_disk(tmp_path), "aquarium", verbose=False)
    assert result.matches == []
    assert result.source == "none"
    assert any("no detector" in n for n in result.notes)


def test_vote_statistics_are_reported_so_min_vote_is_not_a_magic_number(tmp_path):
    match = query_room(room_on_disk(tmp_path), "sofa", verbose=False).matches[0]
    assert set(match.vote_stats) >= {"min_vote", "n_candidates", "n_kept", "kept_frac"}
    assert 0.0 <= match.vote_stats["kept_frac"] <= 1.0


def test_carving_removes_points_the_views_disagree_about(tmp_path):
    match = query_room(room_on_disk(tmp_path), "sofa", verbose=False).matches[0]
    assert match.vote_stats["n_kept"] < match.vote_stats["n_candidates"]


def test_a_match_whose_points_are_all_carved_away_is_reported_as_unsupported(tmp_path):
    """Found in 2D, could not be stood up in 3D. Silence would be a worse answer."""
    result = query_room(
        room_on_disk(tmp_path), "sofa",
        config_overrides={"min_vote": 1.01}, verbose=False,
    )
    assert len(result.matches) == 1
    assert result.matches[0].supported is False
    assert result.matches[0].obb is None


def test_score_is_not_inflated_by_over_carving(tmp_path):
    """`mean_vote` feeds `_score`. If it is averaged over the survivors of the
    `min_vote` cut, raising `min_vote` can only raise it, so an over-carved
    box -- smaller, worse -- reports a *higher* score. Averaged over the
    candidates instead, `mean_vote` (and so `score`) does not move just
    because the cut got stricter."""
    room = room_on_disk(tmp_path)
    loose = query_room(room, "sofa", config_overrides={"min_vote": 0.1}, verbose=False).matches[0]
    tight = query_room(room, "sofa", config_overrides={"min_vote": 0.9}, verbose=False).matches[0]

    assert tight.vote_stats["n_kept"] < loose.vote_stats["n_kept"]   # it did over-carve
    assert tight.score == pytest.approx(loose.score, abs=1e-6)


def test_matches_are_ranked_best_first(tmp_path):
    result = query_room(room_on_disk(tmp_path), "sofa", verbose=False)
    scores = [m.score for m in result.matches]
    assert scores == sorted(scores, reverse=True)


def test_the_result_is_json_serialisable(tmp_path):
    result = query_room(room_on_disk(tmp_path), "sofa", verbose=False)
    json.dumps(result.as_dict())          # must not raise


def test_a_missing_room_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="frames.npz"):
        query_room(tmp_path, "sofa", verbose=False)
