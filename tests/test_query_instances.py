"""Gate 8b: one phrase, several objects.

"chair" matches every chair in the room. Telling them apart uses the same
cross-view vote as carving, because "are these the same object?" is the
question the vote already answers.
"""

import numpy as np

from room3d.artifacts import Reconstruction
from room3d.camera import project_to_frame
from room3d.consensus import View
from room3d.query import group_instances

H, W = 40, 60
K = np.array([[40.0, 0.0, W / 2], [0.0, 40.0, H / 2], [0.0, 0.0, 1.0]])


def camera_at(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def make_recon(xs, depth=100.0):
    ys, px = np.mgrid[0:H, 0:W]
    pts = np.zeros((len(xs), H, W, 3), dtype=np.float32)
    poses = []
    for i, x in enumerate(xs):
        pose = camera_at(x)
        poses.append(pose)
        wx = (px - K[0, 2]) * depth / K[0, 0] + x
        wy = (ys - K[1, 2]) * depth / K[1, 1]
        pts[i] = np.stack([wx, wy, np.full_like(wx, depth)], axis=-1)
    return Reconstruction(
        images=np.zeros((len(xs), H, W, 3), dtype=np.uint8),
        pts3d=pts,
        conf_mask=np.ones((len(xs), H, W), dtype=bool),
        poses=np.asarray(poses, dtype=np.float32),
        intrinsics=np.tile(np.asarray(K, dtype=np.float32), (len(xs), 1, 1)),
        frame_ids=np.arange(len(xs), dtype=np.int32),
    )


def box_around(point, recon, frame, half=5):
    uv, _ = project_to_frame(np.asarray([point], float), recon.poses[frame],
                             recon.intrinsics[frame], recon.image_hw)
    u, v = uv[0]
    return (int(u - half), int(v - half), int(u + half), int(v + half))


def test_repeated_sightings_of_one_object_form_one_instance():
    recon = make_recon([-0.3, 0.0, 0.3])
    target = np.array([0.0, 0.0, 4.0])
    views = [View(i, box_around(target, recon, i), label="chair") for i in range(3)]

    groups = group_instances(views, recon)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_two_separate_objects_form_two_instances():
    """Object A lives in frames 0-1, object B in frames 2-3: no frame is
    shared, so neither side's frames are a subset of the other's and
    `agreement_between` cannot short-circuit on frame overlap (see
    `agreement_between`'s "wholly contained" rule). This has to be true or the
    assertion below would hold no matter where the boxes actually are -- the
    companion test right after this one is the proof that it does not.
    """
    b = 0.15
    recon = make_recon([-3 * b, -b, b, 3 * b])
    left = np.array([-1.2, 0.0, 4.0])
    right = np.array([1.2, 0.0, 4.0])

    views = [View(i, box_around(left, recon, i), label="chair") for i in range(2)]
    views += [View(i, box_around(right, recon, i), label="chair") for i in range(2, 4)]

    assert len(group_instances(views, recon)) == 2


def test_objects_at_the_same_point_merge_into_one_instance():
    """The mirror of the test above, same frame split (0-1 / 2-3) so nothing
    is forced apart by frame overlap -- only the target moved, from two
    distinct points to one. If this did not merge while the separation test
    above still split, the split above would be proof of nothing: a test that
    cannot fail when the thing it guards is broken is not a guard.
    """
    b = 0.15
    recon = make_recon([-3 * b, -b, b, 3 * b])
    same = np.array([0.0, 0.0, 4.0])

    views = [View(i, box_around(same, recon, i), label="chair") for i in range(2)]
    views += [View(i, box_around(same, recon, i), label="chair") for i in range(2, 4)]

    assert len(group_instances(views, recon)) == 1


def test_two_detections_in_the_same_frame_are_always_separate_instances():
    """The detector already decided. Nothing here overrules it."""
    recon = make_recon([0.0])
    views = [
        View(0, (5, 5, 15, 15), label="chair"),
        View(0, (30, 5, 40, 15), label="chair"),
    ]
    assert len(group_instances(views, recon)) == 2


def test_a_lower_threshold_merges_more():
    recon = make_recon([-0.3, 0.0, 0.3])
    left = np.array([-0.6, 0.0, 4.0])
    right = np.array([0.6, 0.0, 4.0])
    views = [View(i, box_around(left, recon, i), label="chair") for i in range(3)]
    views += [View(i, box_around(right, recon, i), label="chair") for i in range(3)]

    strict = len(group_instances(views, recon, min_agreement=0.95))
    loose = len(group_instances(views, recon, min_agreement=0.0))
    assert loose <= strict


def test_no_views_gives_no_instances():
    assert group_instances([], make_recon([0.0])) == []


def test_a_single_view_gives_one_instance():
    recon = make_recon([0.0])
    assert len(group_instances([View(0, (5, 5, 15, 15))], recon)) == 1


def test_instances_are_ordered_by_support():
    """The best-evidenced answer should be match 1, since --commit defaults to it."""
    recon = make_recon([-0.3, 0.0, 0.3])
    left = np.array([-1.2, 0.0, 4.0])
    right = np.array([1.2, 0.0, 4.0])

    views = [View(i, box_around(left, recon, i), label="chair") for i in range(3)]
    views += [View(0, box_around(right, recon, 0), label="chair")]

    sizes = [len(g) for g in group_instances(views, recon)]
    assert sizes == sorted(sizes, reverse=True)
