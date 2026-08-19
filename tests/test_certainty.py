"""Gate 9: losing labels we cannot place, on purpose.

An object whose position is uncertain is worse than a missing one, because a
confident wrong box gets acted on and an absent one does not. What certainty
means here is already computed -- how many views support the match, and how
strongly the surviving points are agreed on -- so this is a filter over
existing evidence, not new machinery.

The governing rule is that nothing vanishes silently: every drop is counted
and returned to the caller.
"""

import json

import numpy as np
import pytest

from room3d.config import LabelConfig, QueryConfig
from room3d.consensus import View
from room3d.projection import OrientedBox
from room3d.query import QueryMatch, commit_match, filter_by_certainty
from room3d.refit import refit_room


def a_match(n_views=3, mean_vote=0.9, supported=True):
    obb = OrientedBox(np.zeros(3), np.ones(3), np.eye(3))
    return QueryMatch(
        label="couch",
        obb=obb if supported else None,
        score=0.8,
        views=[View(i, (0, 0, 10, 10)) for i in range(n_views)],
        n_points=500,
        vote_stats={"mean_vote": mean_vote, "min_vote": 0.6,
                    "n_candidates": 1000, "n_kept": 500, "kept_frac": 0.5},
        supported=supported,
    )


def test_a_single_view_match_is_dropped():
    """One view can be cross-checked against nothing. That IS the uncertain case."""
    kept, dropped = filter_by_certainty([a_match(n_views=1)],
                                        min_views=2, min_mean_vote=0.5)
    assert kept == []
    assert len(dropped) == 1


def test_a_well_supported_match_survives():
    kept, dropped = filter_by_certainty([a_match()], min_views=2, min_mean_vote=0.5)
    assert len(kept) == 1
    assert dropped == []


def test_a_weakly_agreed_match_is_dropped():
    kept, _ = filter_by_certainty([a_match(mean_vote=0.2)],
                                  min_views=2, min_mean_vote=0.5)
    assert kept == []


def test_an_unsupported_match_is_dropped_however_open_the_gate():
    """No 3D points survived carving, so there is no position to be certain of."""
    kept, _ = filter_by_certainty([a_match(supported=False)],
                                  min_views=0, min_mean_vote=0.0)
    assert kept == []


def test_dropped_matches_are_returned_not_discarded():
    _, dropped = filter_by_certainty([a_match(n_views=1)],
                                     min_views=2, min_mean_vote=0.5)
    assert dropped[0].label == "couch"
    assert len(dropped[0].views) == 1


def test_the_gate_can_be_opened_completely():
    kept, dropped = filter_by_certainty([a_match(n_views=1, mean_vote=0.0)],
                                        min_views=0, min_mean_vote=0.0)
    assert len(kept) == 1 and dropped == []


def test_ranking_is_preserved_among_survivors():
    kept, _ = filter_by_certainty([a_match(n_views=5), a_match(n_views=3)],
                                  min_views=2, min_mean_vote=0.5)
    assert [len(m.views) for m in kept] == [5, 3]


# --- the defaults encode the preference ---------------------------------------


def test_query_defaults_require_cross_view_support():
    assert QueryConfig().min_views == 2
    assert QueryConfig().min_mean_vote == 0.5


def test_refit_defaults_stay_lossless():
    """Dropping most of a room is something the user asks for, not a default."""
    assert LabelConfig().min_observations == 1


# --- committing --------------------------------------------------------------


def test_committing_an_uncertain_match_is_refused(tmp_path):
    (tmp_path / "objects.json").write_text(json.dumps({"room": "r", "objects": []}))
    with pytest.raises(ValueError, match="certainty gate"):
        commit_match(tmp_path, a_match(n_views=1), verbose=False)


def test_force_overrides_the_certainty_gate(tmp_path):
    (tmp_path / "objects.json").write_text(json.dumps({"room": "r", "objects": []}))
    assert commit_match(tmp_path, a_match(n_views=1), force=True,
                        verbose=False)["object_id"]


def test_commit_match_gates_against_the_callers_config_not_a_second_default(tmp_path):
    """A caller that already decided a match is committable -- against its own
    `QueryConfig`, e.g. one loosened by `--config` -- must not have that
    decision re-litigated here against a second, independently-defaulted
    `QueryConfig()` and refused."""
    (tmp_path / "objects.json").write_text(json.dumps({"room": "r", "objects": []}))
    loose = QueryConfig(min_views=1, min_mean_vote=0.0)
    assert commit_match(tmp_path, a_match(n_views=1), config=loose,
                        verbose=False)["object_id"]


def test_commit_match_defaults_to_querys_own_default_config(tmp_path):
    """No `config` passed falls back to `QueryConfig()`, matching the module's
    own default rather than an all-permissive one."""
    (tmp_path / "objects.json").write_text(json.dumps({"room": "r", "objects": []}))
    with pytest.raises(ValueError, match="certainty gate"):
        commit_match(tmp_path, a_match(n_views=1), verbose=False)


# --- the same trade, applied to a whole room ---------------------------------


def test_refit_can_drop_thinly_supported_objects(tmp_path):
    """The user's actual case: most objects in the sample room are single
    sightings."""
    from test_refit import legacy_room

    room = legacy_room(tmp_path)
    lossless = refit_room(room, verbose=False)

    strict = refit_room(
        room, config=LabelConfig(min_mask_points=10, min_observations=99),
        verbose=False,
    )
    assert strict["n_objects"] == 0
    assert strict["n_dropped_uncertain"] == lossless["n_objects"]


def test_refit_reports_nothing_dropped_by_default(tmp_path):
    from test_refit import legacy_room

    assert refit_room(legacy_room(tmp_path), verbose=False)["n_dropped_uncertain"] == 0


# --- the CLI: what gets printed as N is what --commit N writes ---------------
#
# A phrase that matches several physical objects returns several QueryMatches
# of mixed certainty. The number a user reads next to a match and the index
# --commit uses have to be the same list by construction, not by the raw list
# and the filtered list happening to agree in whatever fixture a test reaches
# for. `query_room` is monkeypatched here rather than reconstructed through
# the full detection pipeline, precisely so the raw order can be pinned to the
# alternating dropped/kept/dropped/kept shape that is the minimal case where
# the two lists diverge -- filter_by_certainty itself is exercised for real.


def _named_match(label, n_views):
    obb = OrientedBox(np.zeros(3), np.ones(3), np.eye(3))
    return QueryMatch(
        label=label,
        obb=obb,
        score=0.5,
        views=[View(i, (0, 0, 10, 10)) for i in range(n_views)],
        n_points=100,
        vote_stats={"mean_vote": 0.9, "min_vote": 0.6,
                    "n_candidates": 10, "n_kept": 5, "kept_frac": 0.5},
        supported=True,
    )


def _mixed_certainty_matches():
    """dropped, kept, dropped, kept -- so kept = [B, D], and the raw position
    of D (4th) differs from its kept position (2nd)."""
    return [_named_match("chair-A", 1), _named_match("chair-B", 3),
            _named_match("chair-C", 1), _named_match("chair-D", 3)]


@pytest.fixture
def cli_room(tmp_path, monkeypatch):
    """query_room is monkeypatched in every test below, so nothing here reads
    frames.npz or observations.json -- only objects.json needs to exist, for
    commit_match to have something to write into."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out" / "demo"
    out.mkdir(parents=True)
    (out / "objects.json").write_text(
        json.dumps({"room": "demo", "units": "meters", "objects": []})
    )
    return out


def _patch_query_room(monkeypatch, matches):
    import room3d.query as query_mod

    monkeypatch.setattr(
        query_mod, "query_room",
        lambda *a, **k: query_mod.QueryResult("chair", matches, "cache", []),
    )


def test_commit_n_commits_the_object_printed_as_n(cli_room, monkeypatch, capsys):
    from room3d.cli import main

    _patch_query_room(monkeypatch, _mixed_certainty_matches())

    assert main(["query", "--room", "demo", "chair", "--commit", "2"]) == 0
    out = capsys.readouterr().out

    # Printed numbering runs over the kept list only: B first, D second.
    assert "1. chair-B" in out
    assert "2. chair-D" in out
    assert "2 match(es) hidden" in out

    written = json.loads((cli_room / "objects.json").read_text())["objects"]
    assert len(written) == 1
    assert written[0]["label"] == "chair-D"          # printed "2." -- not chair-B


def test_commit_1_commits_the_first_kept_match_not_the_first_raw_match(
    cli_room, monkeypatch
):
    from room3d.cli import main

    _patch_query_room(monkeypatch, _mixed_certainty_matches())

    assert main(["query", "--room", "demo", "chair", "--commit", "1"]) == 0
    written = json.loads((cli_room / "objects.json").read_text())["objects"]
    assert written[0]["label"] == "chair-B"          # raw match 1 is chair-A, dropped


def test_json_matches_array_position_matches_the_commit_target(cli_room, monkeypatch, capsys):
    """The `--json` payload has to give a machine caller the same guarantee
    the printed text gives a human: `matches[i]` is what `--commit i+1` writes."""
    from room3d.cli import main

    _patch_query_room(monkeypatch, _mixed_certainty_matches())

    assert main(["query", "--room", "demo", "chair", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert [m["label"] for m in payload["matches"]] == ["chair-B", "chair-D"]
    assert [m["label"] for m in payload["hidden"]] == ["chair-A", "chair-C"]
