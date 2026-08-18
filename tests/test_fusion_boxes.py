"""Gate 5b: the fused object box must describe the object, not one lucky frame.

The old `_finalise` built a box from three unrelated sources -- centre from the
mean of the cluster centroids, extent element-wise max'd across two *different*
axis frames, rotation borrowed from whichever single observation happened to be
widest. These tests pin the box to the pooled points instead.
"""

import numpy as np
import pytest

from room3d.fusion import cluster_observations
from room3d.level import UP_VECTOR
from room3d.projection import (
    Observation,
    OrientedBox,
    fit_gravity_aligned_box,
    points_inside_fraction,
)


def slab(x_range, n=1500, seed=0):
    """Points on a slice of one long sofa: full height and depth, part of its length."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 3))
    pts[:, 0] = rng.uniform(*x_range, size=n)
    pts[:, 1] = rng.uniform(0.0, 0.9, size=n)
    pts[:, 2] = rng.uniform(-0.45, 0.45, size=n)
    return pts


def obs_from(frame, label, points, conf=0.9):
    points = np.asarray(points, float)
    return Observation(
        frame_idx=frame,
        label=label,
        vlm_confidence=conf,
        centroid=np.median(points, axis=0),
        obb=fit_gravity_aligned_box(points, UP_VECTOR),
        n_points=len(points),
        support=0.9,
        points=points,
    )


def partial_views_of_a_sofa():
    """Three frames, each seeing a different, overlapping part of a 2.4 m sofa.

    No single view spans it, so a box copied from any one observation is short by
    half a metre or more. Only the union is the sofa.
    """
    return [
        obs_from(0, "sofa", slab((-1.2, 0.1), seed=1)),
        obs_from(1, "sofa", slab((-0.5, 0.7), seed=2)),
        obs_from(2, "sofa", slab((-0.1, 1.2), seed=3)),
    ]


# --- the headline failure ----------------------------------------------------


def test_the_fused_box_contains_the_points_of_every_observation():
    """One end landing on the sofa and the rest floating away is exactly this
    assertion failing."""
    obs = partial_views_of_a_sofa()
    objs = cluster_observations(obs, up=UP_VECTOR, obb_percentile=0.0)
    assert len(objs) == 1

    pooled = np.vstack([o.points for o in obs])
    assert points_inside_fraction(objs[0].obb, pooled) > 0.99


def test_the_fused_box_spans_the_whole_object_not_one_partial_view():
    obs = partial_views_of_a_sofa()
    objs = cluster_observations(obs, up=UP_VECTOR, obb_percentile=0.0)
    assert max(objs[0].obb.extent) == pytest.approx(2.4, abs=0.15)


def test_the_fused_box_is_not_wildly_larger_than_the_object_either():
    obs = partial_views_of_a_sofa()
    objs = cluster_observations(obs, up=UP_VECTOR, obb_percentile=0.0)
    sorted_extent = sorted(objs[0].obb.extent)
    assert sorted_extent[0] == pytest.approx(0.9, abs=0.15)     # height
    assert sorted_extent[1] == pytest.approx(0.9, abs=0.15)     # depth


def test_the_fused_centre_agrees_with_the_box_centre():
    """These two disagreed by half a metre in the old output."""
    objs = cluster_observations(partial_views_of_a_sofa(), up=UP_VECTOR)
    assert np.linalg.norm(objs[0].centroid - objs[0].obb.center) < 0.35


def test_the_fused_box_is_gravity_aligned():
    objs = cluster_observations(partial_views_of_a_sofa(), up=UP_VECTOR)
    assert np.allclose(objs[0].obb.R[:, 1], UP_VECTOR, atol=1e-9)


def test_a_yawed_sofa_keeps_its_true_footprint():
    """Turned 40 degrees to the room, the box must turn with it rather than
    inflate to the axis-aligned envelope."""
    t = np.radians(40.0)
    R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    obs = [
        obs_from(i, "sofa", o.points @ R.T)
        for i, o in enumerate(partial_views_of_a_sofa())
    ]
    objs = cluster_observations(obs, up=UP_VECTOR, obb_percentile=0.0)
    assert max(objs[0].obb.extent) == pytest.approx(2.4, abs=0.15)
    assert min(objs[0].obb.extent) == pytest.approx(0.9, abs=0.15)


# --- an extended object swept across many views is still one object ----------


def test_a_long_sofa_swept_across_six_views_is_one_object():
    """Each view overlaps its neighbours but not the far end.

    Comparing against the *mean* of a cluster's centroids cannot express this:
    once two views have merged, the mean sits between them and the third view is
    measured from a place no observation ever was. The question that matters is
    whether the new view coincides with something already in the cluster.
    """
    starts = np.linspace(-1.2, 0.5, 6)
    obs = [
        obs_from(i, "sofa", slab((x, x + 0.7), seed=i))
        for i, x in enumerate(starts)
    ]
    assert len(cluster_observations(obs, up=UP_VECTOR)) == 1


def test_chaining_does_not_swallow_a_row_of_separate_chairs():
    """The counterweight: nearest-member linkage must not walk down a row of
    chairs turning them into one long chair."""
    obs = [obs_from(i, "chair", cube_at(x=1.1 * i, seed=i)) for i in range(5)]
    assert len(cluster_observations(obs, up=UP_VECTOR)) == 5


def cube_at(x, size=0.5, n=1200, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) - 0.5) * size + np.array([x, 0.25, 0.0])


# --- floor snapping ----------------------------------------------------------


def test_an_object_resting_just_above_the_floor_is_snapped_down_to_it():
    obs = [obs_from(i, "sofa", o.points + np.array([0.0, 0.08, 0.0]))
           for i, o in enumerate(partial_views_of_a_sofa())]
    objs = cluster_observations(
        obs, up=UP_VECTOR, floor_height=0.0, floor_snap_threshold=0.15,
        obb_percentile=0.0,
    )
    bottom = objs[0].obb.center[1] - objs[0].obb.extent[1] / 2
    assert bottom == pytest.approx(0.0, abs=1e-6)


def test_an_object_high_on_a_wall_is_not_dragged_to_the_floor():
    obs = [obs_from(i, "shelf", o.points + np.array([0.0, 1.6, 0.0]))
           for i, o in enumerate(partial_views_of_a_sofa())]
    objs = cluster_observations(
        obs, up=UP_VECTOR, floor_height=0.0, floor_snap_threshold=0.15
    )
    assert objs[0].obb.center[1] > 1.5


def test_floor_snapping_is_off_when_no_floor_height_is_given():
    obs = [obs_from(i, "sofa", o.points + np.array([0.0, 0.08, 0.0]))
           for i, o in enumerate(partial_views_of_a_sofa())]
    objs = cluster_observations(obs, up=UP_VECTOR, obb_percentile=0.0)
    bottom = objs[0].obb.center[1] - objs[0].obb.extent[1] / 2
    assert bottom > 0.05


# --- the pointless path still has to behave ----------------------------------


def make_pointless_obs(frame, label, centroid, extent=(0.5, 0.5, 0.5)):
    centroid = np.asarray(centroid, float)
    return Observation(
        frame_idx=frame,
        label=label,
        vlm_confidence=0.9,
        centroid=centroid,
        obb=OrientedBox(centroid.copy(), np.asarray(extent, float), np.eye(3)),
        n_points=500,
        support=0.9,
    )


def test_without_points_the_box_is_one_real_observation_box_unchanged():
    """Re-fusion from observations.json has no points. It must fall back to a
    box that actually existed, never to a mixture of incompatible frames."""
    obs = [
        make_pointless_obs(0, "sofa", [0.0, 0.0, 0.0], extent=(2.0, 0.9, 0.9)),
        make_pointless_obs(1, "sofa", [0.3, 0.0, 0.0], extent=(1.4, 0.9, 0.9)),
        make_pointless_obs(2, "sofa", [0.6, 0.0, 0.0], extent=(1.1, 0.9, 0.9)),
    ]
    objs = cluster_observations(obs)
    assert len(objs) == 1

    widest = obs[0].obb
    assert np.allclose(objs[0].obb.extent, widest.extent)
    assert np.allclose(objs[0].obb.center, widest.center)
    assert np.allclose(objs[0].obb.R, widest.R)


def test_a_mix_of_points_and_no_points_uses_the_points_it_has():
    with_points = obs_from(0, "sofa", slab((-1.2, -0.3), seed=1))
    without = make_pointless_obs(1, "sofa", [0.0, 0.45, 0.0], extent=(2.0, 0.9, 0.9))
    objs = cluster_observations([with_points, without], up=UP_VECTOR, obb_percentile=0.0)
    assert points_inside_fraction(objs[0].obb, with_points.points) > 0.99
