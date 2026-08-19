"""Gate 8f: the CLI verb.

Read-only unless --commit is given. That is the property that makes a vague
query free to try.
"""

import json

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
