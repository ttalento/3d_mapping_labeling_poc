"""Gate 7: cross-view agreement.

A 2D box is axis-aligned and the object inside it is not, so the box contains
things that are not the object. The fix is other cameras: clutter at a different
depth from the object projects outside the box from a different angle. These
tests pin both what that catches and what it cannot.
"""

import numpy as np
import pytest

from room3d.artifacts import Reconstruction
from room3d.consensus import View, consistent, vote

H, W = 40, 40
K = np.array([[40.0, 0.0, W / 2], [0.0, 40.0, H / 2], [0.0, 0.0, 1.0]])


def camera_at(x, z=0.0):
    """A camera on the x/z plane at height 0, looking down +Z."""
    pose = np.eye(4)
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def make_recon(poses, depth=100.0):
    """Cameras with an effectively infinite backdrop, so nothing occludes.

    Occlusion is tested in test_camera.py; these tests are about agreement.
    """
    n = len(poses)
    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.zeros((n, H, W, 3), dtype=np.float32)
    for i, pose in enumerate(poses):
        x = (xs - K[0, 2]) * depth / K[0, 0] + pose[0, 3]
        y = (ys - K[1, 2]) * depth / K[1, 1]
        pts[i] = np.stack([x, y, np.full_like(x, depth + pose[2, 3])], axis=-1)
    return Reconstruction(
        images=np.zeros((n, H, W, 3), dtype=np.uint8),
        pts3d=pts,
        conf_mask=np.ones((n, H, W), dtype=bool),
        poses=np.asarray(poses, dtype=np.float32).reshape(n, 4, 4),
        intrinsics=np.tile(np.asarray(K, dtype=np.float32), (n, 1, 1)),
        frame_ids=np.arange(n, dtype=np.int32),
    )


def box_around(point, recon, frame, half=6):
    """The pixel box centred on where `point` lands in `frame`."""
    from room3d.camera import project_to_frame

    uv, _ = project_to_frame(np.asarray([point], float), recon.poses[frame],
                             recon.intrinsics[frame], recon.image_hw)
    u, v = uv[0]
    return (int(u - half), int(v - half), int(u + half), int(v + half))


# --- the core behaviour --------------------------------------------------------


def test_a_point_inside_every_box_gets_a_full_vote():
    recon = make_recon([camera_at(-1.0), camera_at(0.0), camera_at(1.0)])
    target = np.array([[0.0, 0.0, 4.0]])
    views = [View(i, box_around(target[0], recon, i)) for i in range(3)]

    agree, testable = vote(target, views, recon)
    assert testable[0] == 3
    assert agree[0] == 3


def test_clutter_at_a_different_depth_is_voted_out_by_the_other_views():
    """The whole idea, in one test. A distractor sits inside the box in frame 1
    but parallax carries it outside the box in frames 0 and 2.

    Baseline 0.5, not the 1.5 used elsewhere in this file: at 1.5 no distractor
    depth nearer than the target is simultaneously inside the flanking cameras'
    field of view and outside their box, given this K and the +-6px box half --
    the two constraints are mutually exclusive at that baseline for any such
    depth. 0.5 is the smallest baseline that keeps both satisfiable.
    """
    recon = make_recon([camera_at(-0.5), camera_at(0.0), camera_at(0.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    distractor = np.array([[0.0, 0.0, 1.5]])       # same ray from frame 1, nearer
    agree, testable = vote(distractor, views, recon)

    assert testable[0] == 3
    assert agree[0] == 1                            # only the frame it hides in
    assert not consistent(distractor, views, recon, min_vote=0.6)[0]
    assert consistent(target[None, :], views, recon, min_vote=0.6)[0]


def test_clutter_inside_the_box_in_every_view_survives():
    """The documented limit. With one camera there is no parallax and nothing
    can be carved. Discovering this in the field would be much worse than
    reading it here."""
    recon = make_recon([camera_at(0.0)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(0, box_around(target, recon, 0))]

    distractor = np.array([[0.0, 0.0, 1.5]])
    assert consistent(distractor, views, recon, min_vote=0.6)[0]


def test_a_point_no_frame_can_see_is_never_consistent():
    recon = make_recon([camera_at(0.0), camera_at(1.0)])
    views = [View(i, (0, 0, W, H)) for i in range(2)]

    behind = np.array([[0.0, 0.0, -5.0]])
    agree, testable = vote(behind, views, recon)
    assert testable[0] == 0
    assert not consistent(behind, views, recon)[0]


# --- min_vote ------------------------------------------------------------------


def test_min_vote_is_monotonic():
    """Raising the bar can only remove points, never add them."""
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    rng = np.random.default_rng(0)
    cloud = target + rng.normal(0, 0.4, (400, 3))

    counts = [
        int(consistent(cloud, views, recon, min_vote=m).sum())
        for m in (0.0, 0.34, 0.67, 1.0)
    ]
    assert counts == sorted(counts, reverse=True)


def test_unanimity_is_available_but_is_not_the_default():
    """It was the default in the first probe and it destroyed a speaker seen by
    two cameras.

    0.5, not 0.6: with leave-one-out a point is judged by at most
    n_views - 1 frames, so 0.6 is unanimity for a 3-view object (ceil(0.6 *
    2) == 2 of 2), the modal case in a real room. 0.5 is the largest value
    that does not impose unanimity there (ceil(0.5 * 2) == 1 of 2)."""
    import inspect

    assert inspect.signature(consistent).parameters["min_vote"].default == 0.5


# --- leave-one-out -------------------------------------------------------------


def test_a_point_does_not_vote_for_itself_when_its_source_frame_is_known():
    """A point harvested from frame 1's box is trivially inside frame 1's box.
    Counting that as verification inflates every score by one frame.

    Baseline 0.5 -- see the note in
    test_clutter_at_a_different_depth_is_voted_out_by_the_other_views."""
    recon = make_recon([camera_at(-0.5), camera_at(0.0), camera_at(0.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    distractor = np.array([[0.0, 0.0, 1.5]])
    source = np.array([1])                          # it came from frame 1

    agree, testable = vote(distractor, views, recon, source=source)
    assert testable[0] == 2                         # frame 1 abstains
    assert agree[0] == 0                            # and it had been the only yes


def test_without_a_source_every_frame_votes():
    recon = make_recon([camera_at(-1.5), camera_at(0.0), camera_at(1.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    _, testable = vote(np.array([[0.0, 0.0, 4.0]]), views, recon)
    assert testable[0] == 3


# --- shapes and degenerate input -----------------------------------------------


def test_empty_points_give_empty_results():
    recon = make_recon([camera_at(0.0)])
    agree, testable = vote(np.zeros((0, 3)), [View(0, (0, 0, W, H))], recon)
    assert agree.shape == (0,)
    assert testable.shape == (0,)


def test_no_views_means_nothing_is_testable():
    recon = make_recon([camera_at(0.0)])
    agree, testable = vote(np.array([[0.0, 0.0, 4.0]]), [], recon)
    assert testable[0] == 0
    assert not consistent(np.array([[0.0, 0.0, 4.0]]), [], recon)[0]


def test_view_carries_the_detection_provenance_it_came_from():
    v = View(3, (1, 2, 3, 4), label="couch", vlm_confidence=0.8,
             observation_id=7, object_id="obj_002")
    assert (v.frame_idx, v.box_px, v.label) == (3, (1, 2, 3, 4), "couch")
    assert (v.vlm_confidence, v.observation_id, v.object_id) == (0.8, 7, "obj_002")


# --- harvesting and carving ----------------------------------------------------

from room3d.consensus import candidate_points, carve, largest_component


def test_candidate_points_gathers_the_pixels_inside_every_box():
    recon = make_recon([camera_at(0.0), camera_at(1.0)])
    views = [View(0, (10, 10, 20, 20)), View(1, (5, 5, 10, 10))]

    points, source = candidate_points(views, recon)
    assert len(points) == 10 * 10 + 5 * 5
    assert (source[: 10 * 10] == 0).all()
    assert (source[10 * 10 :] == 1).all()


def test_candidate_points_skips_low_confidence_pixels():
    recon = make_recon([camera_at(0.0)])
    recon.conf_mask[0, 10:15, 10:20] = False
    points, _ = candidate_points([View(0, (10, 10, 20, 20))], recon)
    assert len(points) == 10 * 10 - 5 * 10


def test_candidate_points_with_no_views_is_empty_but_well_shaped():
    recon = make_recon([camera_at(0.0)])
    points, source = candidate_points([], recon)
    assert points.shape == (0, 3)
    assert source.shape == (0,)


def test_carving_removes_the_distractor_and_keeps_the_object():
    """Baseline 0.5, not the 1.5 used elsewhere in this file: `candidate_points`
    harvests real reconstruction pixels, which here sit on `make_recon`'s far
    backdrop (depth 100) rather than at the target's own depth (4). A box's
    *position* tracks the target's parallax (~10px per unit baseline at depth 4),
    but the backdrop pixels *inside* it barely move between frames (~0.4px per
    unit baseline at depth 100). Two frames' boxes can only ever share backdrop
    content when their position shift is under roughly 12 / 10.4 px, i.e. a
    baseline difference below ~1.15 -- at 1.5 (adjacent) or 3.0 (end to end) that
    is never satisfied, so `agree` is 0 for every candidate and carving can never
    keep anything. 0.5 is the same value Task 2 settled on for an analogous
    defect, and is confirmed here to keep 0 < kept < raw."""
    recon = make_recon([camera_at(-0.5), camera_at(0.0), camera_at(0.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    raw, _ = candidate_points(views, recon)
    result = carve(views, recon, keep_largest=False)
    assert 0 < len(result.points) < len(raw)


def test_carving_reports_where_each_surviving_point_came_from():
    """Callers need this to quote an honest vote fraction, which requires
    leave-one-out, which requires knowing each point's source frame.

    Baseline 0.5 -- see the note in
    test_carving_removes_the_distractor_and_keeps_the_object."""
    recon = make_recon([camera_at(-0.5), camera_at(0.0), camera_at(0.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    result = carve(views, recon, keep_largest=False)
    assert result.source.shape == (len(result.points),)
    assert set(np.unique(result.source)) <= {0, 1, 2}


def test_carving_reports_how_much_it_removed():
    """Baseline 0.5 -- see the note in
    test_carving_removes_the_distractor_and_keeps_the_object."""
    recon = make_recon([camera_at(-0.5), camera_at(0.0), camera_at(0.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    stats = carve(views, recon, keep_largest=False).stats
    assert stats["n_kept"] < stats["n_candidates"]
    assert 0.0 <= stats["kept_frac"] <= 1.0
    assert 0.0 <= stats["mean_vote"] <= 1.0
    assert stats["min_vote"] == 0.5              # carve's own default; see
    # test_unanimity_is_available_but_is_not_the_default for why 0.5


def test_mean_vote_is_computed_over_candidates_not_survivors():
    """`min_vote` keeps points whose fraction already clears the bar, so a
    `mean_vote` averaged over the survivors is `>= min_vote` by construction --
    raising the bar can only raise it further, rewarding over-carving with a
    *higher* reported confidence. Averaged over the full candidate set instead,
    it says how agreed the object is regardless of where the bar was set.

    Baseline 0.5 -- see the note in
    test_carving_removes_the_distractor_and_keeps_the_object."""
    recon = make_recon([camera_at(-0.5), camera_at(0.0), camera_at(0.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    loose = carve(views, recon, min_vote=0.1, keep_largest=False)
    tight = carve(views, recon, min_vote=0.9, keep_largest=False)

    assert tight.stats["n_kept"] < loose.stats["n_kept"]              # still over-carves
    assert tight.stats["mean_vote"] == pytest.approx(loose.stats["mean_vote"])


def test_the_survivor_mean_vote_is_still_reported_under_its_own_key():
    """The old, kept-only number is a real (if construction-biased) quantity
    and stays available -- under a name that does not collide with the one
    `_score`/`filter_by_certainty` now consume."""
    recon = make_recon([camera_at(-0.5), camera_at(0.0), camera_at(0.5)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i)) for i in range(3)]

    loose = carve(views, recon, min_vote=0.1, keep_largest=False)
    tight = carve(views, recon, min_vote=0.9, keep_largest=False)

    assert "kept_mean_vote" in loose.stats
    assert tight.stats["kept_mean_vote"] >= loose.stats["kept_mean_vote"]


def test_carving_a_single_view_returns_its_candidates_unchanged():
    """Leave-one-out leaves nothing to check against. Returning an empty result
    would read as 'not there' when the truth is 'not verifiable'."""
    recon = make_recon([camera_at(0.0)])
    views = [View(0, (10, 10, 20, 20))]

    raw, _ = candidate_points(views, recon)
    result = carve(views, recon)
    assert len(result.points) == len(raw)
    assert result.stats["mean_vote"] == 0.0        # nothing verified it


def test_largest_component_keeps_the_bigger_of_two_separated_blobs():
    rng = np.random.default_rng(0)
    big = rng.normal(0, 0.05, (500, 3))
    small = rng.normal(0, 0.05, (50, 3)) + np.array([3.0, 0.0, 0.0])
    points = np.vstack([big, small])

    keep = largest_component(points, voxel=0.05)
    assert keep[:500].all()
    assert not keep[500:].any()


def test_largest_component_keeps_everything_when_it_is_one_blob():
    rng = np.random.default_rng(0)
    points = rng.normal(0, 0.05, (300, 3))
    assert largest_component(points, voxel=0.05).all()


def test_largest_component_coarsens_rather_than_allocating_a_vast_grid():
    """A single far outlier must not turn a 4 cm grid into gigabytes."""
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 0.05, (200, 3)), [[500.0, 500.0, 500.0]]])
    keep = largest_component(points, voxel=0.001, max_cells=100_000)
    assert keep.shape == (201,)


def test_largest_component_handles_too_few_points_to_cluster():
    assert largest_component(np.zeros((1, 3))).all()
    assert largest_component(np.zeros((0, 3))).shape == (0,)


# --- the identity test ---------------------------------------------------------

from room3d.consensus import agreement_between

# Baseline 0.15, not the 1.5 used in the brief. `make_recon`'s backdrop is flat
# at depth 100 -- there is no geometry at the target's own depth (4), so
# `candidate_points` harvests backdrop pixels from inside a box that was
# centred on where the *target* projects, not on anything actually at that
# depth. Reprojecting those backdrop points into the other frame lands them
# offset from that frame's own target-centred box by
# `fx * (c1 - c0) * (1/Z_target - 1/depth)` = `9.6 * (c1 - c0)` px for this
# K and target depth -- the backdrop parallax (depth 100) and the box's own
# parallax (depth 4) disagree, and the mismatch grows with baseline. At 1.5
# (c1 - c0 = 3.0) that is ~29px against a 12px box: zero overlap, so agreement
# is always 0 regardless of implementation. At 0.5, the baseline the same
# defect was fixed with elsewhere in this file, it is still ~10px: only ~13%
# overlap, well under the 0.7 bar. Overlap only clears 0.7 below baseline
# ~0.19; 0.15 lands mid-plateau there (min direction = 9/12 = 0.75) with
# margin on both sides -- confirmed against the real `candidate_points`/`vote`
# pair before writing this fixture, not assumed.
_IDENTITY_BASELINE = 0.15


def test_two_views_of_the_same_object_agree_strongly():
    recon = make_recon([camera_at(-_IDENTITY_BASELINE), camera_at(_IDENTITY_BASELINE)])
    target = np.array([0.0, 0.0, 4.0])

    a = [View(0, box_around(target, recon, 0))]
    b = [View(1, box_around(target, recon, 1))]
    assert agreement_between(a, b, recon) > 0.7


def test_views_of_two_different_objects_do_not_agree():
    recon = make_recon([camera_at(-_IDENTITY_BASELINE), camera_at(_IDENTITY_BASELINE)])
    left = np.array([-1.0, 0.0, 4.0])
    right = np.array([1.5, 0.0, 4.0])

    a = [View(0, box_around(left, recon, 0))]
    b = [View(1, box_around(right, recon, 1))]
    assert agreement_between(a, b, recon) < 0.3


def test_agreement_is_symmetric():
    recon = make_recon([camera_at(-_IDENTITY_BASELINE), camera_at(_IDENTITY_BASELINE)])
    target = np.array([0.0, 0.0, 4.0])
    a = [View(0, box_around(target, recon, 0))]
    b = [View(1, box_around(target, recon, 1))]

    assert agreement_between(a, b, recon) == pytest.approx(
        agreement_between(b, a, recon)
    )


def test_a_small_object_inside_a_large_one_is_not_merged_with_it():
    """Every speaker point is inside the cabinet's box, so the speaker->cabinet
    direction scores high (measured 1.0). The reverse does not (measured
    ~0.020) -- most of the cabinet's points are nowhere near the speaker's
    small box -- and it is the **minimum** of the two that stops a speaker
    being absorbed into the cabinet it happens to sit inside.

    The threshold is 0.3, not the 0.6 in the brief: at 0.6, `min` (~0.02) and
    `mean` (~0.51) of the two directional scores both pass, so a regression
    from min to mean would go undetected. 0.3 sits with real margin below the
    min score and real margin above the mean score, so it fails the moment
    the minimum is replaced by anything that averages the two directions."""
    recon = make_recon([camera_at(-_IDENTITY_BASELINE), camera_at(_IDENTITY_BASELINE)])
    target = np.array([0.0, 0.0, 4.0])

    large = [View(0, box_around(target, recon, 0, half=14))]
    small = [View(1, box_around(target, recon, 1, half=2))]
    assert agreement_between(small, large, recon) < 0.3


def test_two_boxes_in_the_same_frame_never_merge():
    """The detector already said these are two objects. Leave-one-out makes that
    conclusion fall out rather than needing a special case."""
    recon = make_recon([camera_at(0.0)])
    a = [View(0, (5, 5, 15, 15))]
    b = [View(0, (6, 6, 16, 16))]
    assert agreement_between(a, b, recon) == 0.0


def test_agreement_with_an_empty_side_is_zero():
    recon = make_recon([camera_at(0.0)])
    assert agreement_between([], [View(0, (5, 5, 15, 15))], recon) == 0.0


# --- the free mask -------------------------------------------------------------

from room3d.consensus import consistency_mask

# Baseline 1.0, not the 1.5 in the brief. `consistency_mask` harvests real
# reconstruction pixels from frame 1's *own* box -- backdrop points at depth
# 100 -- then asks frames 0 and 2, whose boxes are centred on the target's own
# depth (4), whether each one lands inside. A column at offset `delta` from
# frame 1's own box centre (delta in [-half, half]) reprojects into a flanking
# box at offset `9.6 * baseline + delta` from that box's centre (backdrop
# parallax 0.4 px/unit baseline vs. the box's own 10 px/unit baseline; see the
# analogous derivation on `test_carving_removes_the_distractor_and_keeps_the_object`
# for the same 9.6 constant). Both flanking boxes can agree on the same column
# only where the two offset windows overlap, which needs baseline <= half/9.6.
# For half=12 that is 1.25; at 1.5 no column is ever doubly confirmed and the
# mask would always be empty regardless of implementation. 1.0 leaves a margin
# under 1.25 and confirms (against the real `vote`/`consistent` pair) that it
# yields a non-empty, non-full column band -- so the mask is a genuine strict
# subset, not merely trivially so.
_MASK_BASELINE = 1.0


def test_the_mask_is_a_subset_of_the_box_it_explains():
    """half=12, not the default half=6: at half=6 the derivation above (see
    `_MASK_BASELINE`) puts the mask at 0 of 144 pixels, so `not outside.any()`
    would hold no matter what the implementation did -- a vacuous pass. At
    half=12 it is 120 of 576, a genuine strict subset, and non-emptiness is
    checked directly below rather than assumed."""
    recon = make_recon([camera_at(-_MASK_BASELINE), camera_at(0.0), camera_at(_MASK_BASELINE)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i, half=12)) for i in range(3)]

    mask = consistency_mask(1, views, recon)
    assert mask.shape == recon.image_hw
    assert mask.any()                             # non-vacuous: something was kept

    x0, y0, x1, y1 = views[1].box_px
    outside = mask.copy()
    outside[y0:y1, x0:x1] = False
    assert not outside.any()


def test_the_mask_marks_fewer_pixels_than_the_box_contains():
    """If it marked all of them it would be telling you nothing."""
    recon = make_recon([camera_at(-_MASK_BASELINE), camera_at(0.0), camera_at(_MASK_BASELINE)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i, half=12)) for i in range(3)]

    x0, y0, x1, y1 = views[1].box_px
    assert consistency_mask(1, views, recon).sum() < (x1 - x0) * (y1 - y0)


def test_a_frame_with_no_box_gets_an_empty_mask():
    recon = make_recon([camera_at(0.0), camera_at(1.0)])
    mask = consistency_mask(1, [View(0, (5, 5, 15, 15))], recon)
    assert not mask.any()


def test_raising_min_vote_can_only_shrink_the_mask():
    recon = make_recon([camera_at(-_MASK_BASELINE), camera_at(0.0), camera_at(_MASK_BASELINE)])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i, half=12)) for i in range(3)]

    loose = consistency_mask(1, views, recon, min_vote=0.0).sum()
    tight = consistency_mask(1, views, recon, min_vote=1.0).sum()
    assert tight <= loose
