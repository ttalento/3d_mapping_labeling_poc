"""Gate 5: fusion must merge repeats without collapsing distinct objects.

Both failure directions are tested, because they pull against each other and a
change that fixes one usually breaks the other.
"""

import numpy as np
import pytest

from room3d.fusion import (
    aabb_iou,
    cluster_observations,
    default_canonicalize,
    default_label_compatible,
    normalize_label,
)
from room3d.projection import Observation, OrientedBox


def make_obs(frame, label, centroid, extent=(0.5, 0.5, 0.5), conf=0.9, n_points=500):
    centroid = np.asarray(centroid, dtype=float)
    return Observation(
        frame_idx=frame,
        label=label,
        vlm_confidence=conf,
        centroid=centroid,
        obb=OrientedBox(centroid.copy(), np.asarray(extent, float), np.eye(3)),
        n_points=n_points,
        support=0.9,
    )


# --- the headline test -------------------------------------------------------


def test_five_views_of_one_object_plus_one_other_gives_two_clusters():
    """Gate 5, exactly as specified in the plan."""
    rng = np.random.default_rng(0)
    obs = [
        make_obs(i, "chair", np.array([1.0, 0.0, 2.0]) + rng.normal(0, 0.04, 3))
        for i in range(5)
    ]
    obs.append(make_obs(5, "desk", [3.0, 0.0, 2.0]))

    objs = cluster_observations(obs)
    assert len(objs) == 2

    by_label = {o.label: o for o in objs}
    assert by_label["chair"].n_observations == 5
    assert by_label["desk"].n_observations == 1
    assert by_label["chair"].seen_in == [0, 1, 2, 3, 4]


def test_same_label_far_apart_stays_separate():
    """Four chairs around a table are four objects, not one."""
    obs = [
        make_obs(0, "chair", [0.0, 0.0, 0.0]),
        make_obs(1, "chair", [2.5, 0.0, 0.0]),
        make_obs(2, "chair", [0.0, 0.0, 2.5]),
        make_obs(3, "chair", [2.5, 0.0, 2.5]),
    ]
    assert len(cluster_observations(obs)) == 4


def test_different_labels_at_same_place_stay_separate():
    """A monitor sitting on a desk shares space but is not the desk."""
    obs = [
        make_obs(0, "monitor", [1.0, 0.0, 2.0]),
        make_obs(1, "desk", [1.02, 0.0, 2.0]),
    ]
    labels = {o.label for o in cluster_observations(obs)}
    assert labels == {"monitor", "desk"}


def test_synonyms_merge_and_are_recorded_as_aliases():
    obs = [
        make_obs(0, "monitor", [1.0, 0.0, 2.0]),
        make_obs(1, "computer screen", [1.03, 0.0, 2.0]),
        make_obs(2, "display", [0.98, 0.0, 2.0]),
    ]
    objs = cluster_observations(obs)
    assert len(objs) == 1
    assert objs[0].label == "monitor"          # most frequent after normalisation
    assert set(objs[0].aliases) == {"computer screen", "display"}


# --- size-adaptive radius: the reason a flat threshold fails ------------------


def test_large_objects_tolerate_larger_centroid_disagreement():
    """Partial views of a sofa put centroids far apart; they are still one sofa."""
    big = (2.2, 0.9, 0.9)
    obs = [
        make_obs(0, "sofa", [0.0, 0.0, 0.0], extent=big),
        make_obs(1, "sofa", [0.7, 0.0, 0.0], extent=big),
    ]
    assert len(cluster_observations(obs)) == 1


def test_small_objects_at_the_same_separation_stay_separate():
    """The identical 0.7 m gap must split two mugs. Same distance, different verdict
    -- this is precisely what a flat threshold cannot express."""
    small = (0.09, 0.12, 0.09)
    obs = [
        make_obs(0, "mug", [0.0, 0.0, 0.0], extent=small),
        make_obs(1, "mug", [0.7, 0.0, 0.0], extent=small),
    ]
    assert len(cluster_observations(obs)) == 2


# --- confidence --------------------------------------------------------------


def test_more_observations_yield_higher_confidence():
    many = cluster_observations(
        [make_obs(i, "chair", [1.0, 0.0, 2.0]) for i in range(5)]
    )
    one = cluster_observations([make_obs(0, "chair", [1.0, 0.0, 2.0])])
    assert many[0].confidence > one[0].confidence


def test_scattered_observations_yield_lower_confidence_than_tight_ones():
    tight = cluster_observations(
        [make_obs(i, "chair", [1.0, 0.0, 2.0] + np.array([0.01 * i, 0, 0])) for i in range(4)]
    )
    loose = cluster_observations(
        [make_obs(i, "chair", [1.0, 0.0, 2.0] + np.array([0.12 * i, 0, 0])) for i in range(4)]
    )
    assert len(tight) == 1 and len(loose) == 1
    assert tight[0].confidence > loose[0].confidence


def test_low_vlm_confidence_lowers_object_confidence():
    sure = cluster_observations([make_obs(i, "chair", [1, 0, 2], conf=0.95) for i in range(3)])
    unsure = cluster_observations([make_obs(i, "chair", [1, 0, 2], conf=0.30) for i in range(3)])
    assert sure[0].confidence > unsure[0].confidence


def test_confidence_stays_in_unit_range():
    for objs in (
        cluster_observations([make_obs(i, "chair", [1, 0, 2], conf=1.0) for i in range(20)]),
        cluster_observations([make_obs(0, "chair", [1, 0, 2], conf=0.01)]),
    ):
        assert 0.0 <= objs[0].confidence <= 1.0


# --- output shape ------------------------------------------------------------


def test_objects_are_ordered_by_observation_count():
    obs = [make_obs(i, "chair", [0, 0, 0]) for i in range(4)]
    obs += [make_obs(i, "lamp", [5, 0, 0]) for i in range(2)]
    obs += [make_obs(0, "plant", [10, 0, 0])]
    counts = [o.n_observations for o in cluster_observations(obs)]
    assert counts == sorted(counts, reverse=True)


def test_ids_are_unique_and_stable_format():
    obs = [make_obs(0, "chair", [0, 0, 0]), make_obs(1, "lamp", [5, 0, 0])]
    ids = [o.id for o in cluster_observations(obs)]
    assert ids == ["obj_000", "obj_001"]


def test_as_dict_is_json_serialisable():
    import json
    objs = cluster_observations([make_obs(0, "chair", [1, 2, 3])])
    json.dumps(objs[0].as_dict())        # must not raise


def test_empty_input_gives_empty_output():
    assert cluster_observations([]) == []


def test_injected_label_compatible_is_respected():
    """The CrewAI layer swaps in an LLM adjudicator; it must actually be used."""
    obs = [
        make_obs(0, "wombat", [1.0, 0.0, 2.0]),
        make_obs(1, "quokka", [1.02, 0.0, 2.0]),
    ]
    assert len(cluster_observations(obs)) == 2
    assert len(cluster_observations(obs, label_compatible=lambda a, b: True)) == 1


def test_injected_canonicalize_is_respected():
    obs = [make_obs(0, "chair", [1, 0, 2])]
    objs = cluster_observations(obs, canonicalize=lambda labels: "SEATING")
    assert objs[0].label == "SEATING"


# --- helpers -----------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("Monitor", "monitor", True),
        ("computer_screen", "computer screen", True),
        ("monitor", "display", True),
        ("chair", "desk", False),
        ("  Sofa ", "couch", True),
    ],
)
def test_default_label_compatible(a, b, expected):
    assert default_label_compatible(a, b) is expected


def test_normalize_label_collapses_whitespace_and_case():
    assert normalize_label("  Computer_Screen  ") == "computer screen"


def test_default_canonicalize_picks_most_frequent():
    assert default_canonicalize(["monitor", "display", "monitor"]) == "monitor"


def test_default_canonicalize_breaks_ties_with_synonym_primary():
    """'display' and 'monitor' are both 7 chars and both seen once, so length and
    alphabetical order cannot decide. The synonym table's primary name does."""
    assert default_canonicalize(["display", "monitor"]) == "monitor"
    assert default_canonicalize(["couch", "sofa"]) == "sofa"


def test_default_canonicalize_breaks_remaining_ties_by_length():
    assert default_canonicalize(["potted plant", "plant"]) == "plant"


def test_default_canonicalize_frequency_beats_synonym_primary():
    assert default_canonicalize(["display", "display", "monitor"]) == "display"


def test_aabb_iou_identical_boxes_is_one():
    box = OrientedBox(np.zeros(3), np.array([1.0, 1.0, 1.0]), np.eye(3))
    assert aabb_iou(box, box) == pytest.approx(1.0)


def test_aabb_iou_disjoint_boxes_is_zero():
    a = OrientedBox(np.zeros(3), np.ones(3), np.eye(3))
    b = OrientedBox(np.array([10.0, 0, 0]), np.ones(3), np.eye(3))
    assert aabb_iou(a, b) == 0.0
