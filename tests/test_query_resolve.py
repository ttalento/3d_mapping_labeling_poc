"""Gate 8a: turning a phrase into the boxes already on disk.

The expensive part of a run is the 2D detections. A query that can answer from
them costs nothing, so this is the path that must be tried first.
"""

import pytest

from room3d.config import Config, QueryConfig, load_config
from room3d.query import cached_views, normalize_phrase


def doc(*entries):
    return {
        "observations": [
            {
                "id": i,
                "object_id": obj,
                "frame_idx": frame,
                "label": label,
                "vlm_confidence": conf,
                "box_px": [1, 2, 3, 4],
            }
            for i, (obj, frame, label, conf) in enumerate(entries)
        ]
    }


SAMPLE = doc(
    ("obj_000", 0, "couch", 0.9),
    ("obj_000", 1, "sofa", 0.8),
    ("obj_001", 2, "chair", 0.7),
    ("obj_002", 3, "chair", 0.6),
)


# --- normalisation -------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("Couch", "couch"),
        ("  the couch  ", "couch"),
        ("A chair", "chair"),
        ("an office chair", "office chair"),
        ("the couch by the window", "couch by the window"),
        ("office_chair", "office chair"),
    ],
)
def test_normalize_phrase(phrase, expected):
    assert normalize_phrase(phrase) == expected


def test_only_a_leading_article_is_stripped():
    """'the' inside the phrase is part of a qualifier and must survive, or the
    qualifier silently disappears and the wrong question gets answered."""
    assert normalize_phrase("the sofa near the door") == "sofa near the door"


# --- matching ------------------------------------------------------------------


def test_a_plain_label_matches_its_detections():
    views = cached_views(SAMPLE, "couch")
    assert {v.frame_idx for v in views} == {0, 1}


def test_synonyms_match_through_the_existing_table():
    """couch and sofa are one object; fusion already knows that."""
    assert {v.frame_idx for v in cached_views(SAMPLE, "sofa")} == {0, 1}


def test_a_label_shared_by_several_objects_returns_all_of_them():
    """Splitting them into instances is a later stage's job, not this one's."""
    views = cached_views(SAMPLE, "chair")
    assert {v.frame_idx for v in views} == {2, 3}
    assert {v.object_id for v in views} == {"obj_001", "obj_002"}


def test_a_qualified_phrase_matches_nothing_so_it_can_fall_through_to_the_vlm():
    assert cached_views(SAMPLE, "the couch by the window") == []


def test_an_unknown_label_matches_nothing():
    assert cached_views(SAMPLE, "aquarium") == []


def test_views_carry_the_provenance_a_commit_will_need():
    view = cached_views(SAMPLE, "couch")[0]
    assert view.observation_id == 0
    assert view.object_id == "obj_000"
    assert view.label == "couch"
    assert view.vlm_confidence == pytest.approx(0.9)
    assert view.box_px == (1, 2, 3, 4)


def test_observations_without_a_pixel_box_are_skipped():
    """Nothing to project. Carrying it forward would fake evidence."""
    d = doc(("obj_000", 0, "couch", 0.9))
    d["observations"][0]["box_px"] = None
    assert cached_views(d, "couch") == []


def test_an_injected_matcher_is_respected():
    """The CrewAI layer swaps in an LLM adjudicator, exactly as fusion does."""
    assert len(cached_views(SAMPLE, "wombat", label_compatible=lambda a, b: True)) == 4


# --- config --------------------------------------------------------------------


def test_query_config_has_the_documented_defaults():
    c = QueryConfig()
    assert c.min_vote == 0.6
    assert c.occlusion_tol == 0.10
    assert c.keep_largest_component is True
    assert c.component_voxel == 0.04
    assert c.min_instance_agreement == 0.3


def test_query_config_is_reachable_from_the_top_level_config():
    assert isinstance(Config().query, QueryConfig)


def test_query_config_loads_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("query:\n  min_vote: 0.8\n")
    assert load_config(path).query.min_vote == 0.8


def test_unknown_query_keys_are_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("query:\n  nonsense: 1\n")
    with pytest.raises(ValueError, match="nonsense"):
        load_config(path)
