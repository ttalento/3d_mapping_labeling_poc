"""Gate 8f: the CLI verb.

Read-only unless --commit is given. That is the property that makes a vague
query free to try.
"""

import json

import numpy as np
import pytest

from room3d.artifacts import save_frames_npz
from room3d.cli import main
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from test_boxfit_wiring import labelled_session


@pytest.fixture
def room(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out" / "demo"
    out.mkdir(parents=True)

    session = labelled_session()
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)

    save_frames_npz(out / "frames.npz", session.recon)
    (out / "observations.json").write_text(
        json.dumps(build_observations_doc(session, "demo"))
    )
    (out / "objects.json").write_text(
        json.dumps({"room": "demo", "units": "meters", "scale_verified": False,
                    "n_frames_labeled": 4,
                    "objects": [o.as_dict() for o in session.objects]})
    )
    return out


def test_query_returns_zero_and_prints_a_match(room, capsys):
    assert main(["query", "--room", "demo", "sofa"]) == 0
    assert "sofa" in capsys.readouterr().out


def test_query_does_not_modify_the_room(room):
    before = (room / "objects.json").read_text()
    main(["query", "--room", "demo", "sofa"])
    assert (room / "objects.json").read_text() == before


def test_json_output_parses(room, capsys):
    main(["query", "--room", "demo", "sofa", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["phrase"] == "sofa"
    assert payload["matches"]


def test_commit_writes_the_match(room):
    before = (room / "objects.json").read_text()
    assert main(["query", "--room", "demo", "sofa", "--commit", "1"]) == 0
    assert (room / "objects.json").read_text() != before
    assert (room / "objects.prev.json").exists()


def test_commit_is_one_indexed_and_rejects_a_bad_index(room, capsys):
    assert main(["query", "--room", "demo", "sofa", "--commit", "0"]) == 1
    assert "1-indexed" in capsys.readouterr().err


def test_a_missing_room_reports_an_error_rather_than_a_traceback(room, capsys):
    assert main(["query", "--room", "nope", "sofa"]) == 1
    assert "not found" in capsys.readouterr().err


def test_min_vote_can_be_overridden_from_the_command_line(room, capsys):
    main(["query", "--room", "demo", "sofa", "--min-vote", "0.9", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["matches"][0]["vote_stats"]["min_vote"] == 0.9


def test_no_match_is_reported_and_is_not_an_error(room, capsys):
    assert main(["query", "--room", "demo", "aquarium"]) == 0
    assert "no detector" in capsys.readouterr().out


# --- a VLM match can never absorb -- finding 2 must make that visible --------


def test_force_help_warns_that_vlm_matches_cannot_absorb(capsys):
    with pytest.raises(SystemExit):
        main(["query", "--help"])
    out = capsys.readouterr().out
    assert "absorb" in out


def _vlm_match(label="couch"):
    from room3d.consensus import View
    from room3d.projection import OrientedBox
    from room3d.query import QueryMatch

    obb = OrientedBox(np.zeros(3), np.ones(3), np.eye(3))
    return QueryMatch(
        label=label,
        obb=obb,
        score=0.8,
        # A VLM-sourced View, exactly as `_vlm_views` builds one: no
        # observation_id, no object_id.
        views=[View(i, (0, 0, 10, 10), label=label, vlm_confidence=0.85)
               for i in range(2)],
        n_points=100,
        vote_stats={"mean_vote": 0.9, "min_vote": 0.5,
                    "n_candidates": 100, "n_kept": 100, "kept_frac": 1.0},
        absorbed_object_ids=[],
        supported=True,
    )


@pytest.fixture
def vlm_room(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out" / "demo"
    out.mkdir(parents=True)
    (out / "objects.json").write_text(
        json.dumps({"room": "demo", "units": "meters", "objects": []})
    )
    return out


def _patch_vlm_query_room(monkeypatch, matches):
    import room3d.query as query_mod

    monkeypatch.setattr(
        query_mod, "query_room",
        lambda *a, **k: query_mod.QueryResult("the couch by the window", matches, "vlm", []),
    )


def test_committing_a_vlm_match_that_absorbs_nothing_says_so(vlm_room, monkeypatch, capsys):
    _patch_vlm_query_room(monkeypatch, [_vlm_match()])

    assert main([
        "query", "--room", "demo", "the couch by the window", "--commit", "1",
    ]) == 0
    out = capsys.readouterr().out
    assert "added" in out and "not" in out


def test_committing_a_cached_match_that_absorbs_nothing_says_nothing_extra(
    room, monkeypatch, capsys
):
    """The note is specific to the VLM path -- a cache-sourced match with
    nothing to absorb is the ordinary "fresh object" case, not a limitation."""
    import room3d.query as query_mod

    match = _vlm_match()
    monkeypatch.setattr(
        query_mod, "query_room",
        lambda *a, **k: query_mod.QueryResult("couch", [match], "cache", []),
    )

    assert main(["query", "--room", "demo", "couch", "--commit", "1"]) == 0
    out = capsys.readouterr().out
    assert "VLM" not in out
