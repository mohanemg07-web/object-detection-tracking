"""Unit tests for the ByteTrack core (geometry, association, tracking)."""

import numpy as np

from src.tracking.bytetrack import (
    ByteTracker,
    Track,
    cxcyah_to_xyxy,
    iou_matrix,
    linear_assignment,
    xyxy_to_cxcyah,
)


def test_xyxy_cxcyah_roundtrip():
    box = np.array([10.0, 20.0, 50.0, 100.0])  # w=40, h=80
    state = xyxy_to_cxcyah(box)
    assert state[0] == 30.0  # cx
    assert state[1] == 60.0  # cy
    assert state[2] == 0.5  # aspect w/h = 40/80
    assert state[3] == 80.0  # h
    back = cxcyah_to_xyxy(state)
    assert np.allclose(back, box)


def test_iou_matrix_identical_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    iou = iou_matrix(a, a)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == 1.0


def test_iou_matrix_disjoint_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_matrix_half_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])  # area 100
    b = np.array([[5.0, 0.0, 15.0, 10.0]])  # area 100, overlap 50
    # IoU = 50 / (100 + 100 - 50) = 50/150 = 1/3
    assert np.isclose(iou_matrix(a, b)[0, 0], 1.0 / 3.0)


def test_iou_matrix_empty_inputs():
    assert iou_matrix(np.empty((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_linear_assignment_perfect_diagonal():
    # cost 0 on diagonal, high elsewhere -> identity matching
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert ua == [] and ub == []


def test_linear_assignment_respects_threshold():
    # all costs above threshold -> no matches
    cost = np.array([[0.9, 0.9], [0.9, 0.9]])
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert matches == []
    assert sorted(ua) == [0, 1]
    assert sorted(ub) == [0, 1]


def test_linear_assignment_empty():
    matches, ua, ub = linear_assignment(np.empty((0, 0)), thresh=0.5)
    assert matches == []


def test_tracker_creates_and_maintains_track_id():
    Track.reset_ids()
    tracker = ByteTracker({"new_track_thresh": 0.3, "track_high_thresh": 0.4})
    boxes = np.array([[10.0, 10.0, 50.0, 90.0]])
    scores = np.array([0.9])
    classes = np.array([3])

    # first frame spawns a tentative track (not yet confirmed/returned)
    tracker.update(boxes, scores, classes)
    # second frame with a matching detection confirms it
    active = tracker.update(boxes + 2, scores, classes)
    assert len(active) == 1
    first_id = active[0].track_id

    # third frame: same object should keep the same id
    active = tracker.update(boxes + 4, scores, classes)
    assert len(active) == 1
    assert active[0].track_id == first_id


def test_tracker_low_conf_second_association():
    """A track should survive a frame where the detection drops to low conf."""
    Track.reset_ids()
    tracker = ByteTracker(
        {"track_high_thresh": 0.5, "track_low_thresh": 0.1, "new_track_thresh": 0.5}
    )
    box = np.array([[10.0, 10.0, 50.0, 90.0]])
    # confirm the track over two high-conf frames
    tracker.update(box, np.array([0.9]), np.array([3]))
    active = tracker.update(box, np.array([0.9]), np.array([3]))
    tid = active[0].track_id
    # now a low-confidence detection: stage-2 association should keep the id
    active = tracker.update(box, np.array([0.2]), np.array([3]))
    assert len(active) == 1
    assert active[0].track_id == tid


def test_tracker_drops_stale_track_after_buffer():
    Track.reset_ids()
    tracker = ByteTracker({"track_high_thresh": 0.5, "new_track_thresh": 0.5, "track_buffer": 3})
    box = np.array([[10.0, 10.0, 50.0, 90.0]])
    tracker.update(box, np.array([0.9]), np.array([3]))
    tracker.update(box, np.array([0.9]), np.array([3]))  # confirmed

    empty = np.empty((0, 4))
    es = np.empty((0,))
    ec = np.empty((0,), dtype=int)
    # feed empty frames beyond track_buffer
    for _ in range(5):
        active = tracker.update(empty, es, ec)
    assert active == []
    assert tracker.tracks == []
