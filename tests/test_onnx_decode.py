"""Tests for the ONNX detector decode logic (NMS + decode), no model needed."""

import numpy as np

from src.inference.onnx_detector import Detections, decode_predictions, nms


def test_nms_suppresses_overlapping_boxes():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # heavily overlaps the first
            [100.0, 100.0, 110.0, 110.0],  # far away
        ]
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thresh=0.5)
    # the second box is suppressed; first and third survive
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_nms_empty():
    assert nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []


def test_nms_keeps_all_when_disjoint():
    boxes = np.array([[0, 0, 5, 5], [10, 10, 15, 15], [20, 20, 25, 25]], dtype=float)
    scores = np.array([0.5, 0.6, 0.7])
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert sorted(keep) == [0, 1, 2]


def _fake_yolo_output(cx, cy, w, h, cls, nc=10, conf=0.9):
    """Build a (1, 4+nc, 1) YOLOv8-style output with one anchor."""
    out = np.zeros((1, 4 + nc, 1), dtype=np.float32)
    out[0, 0, 0] = cx
    out[0, 1, 0] = cy
    out[0, 2, 0] = w
    out[0, 3, 0] = h
    out[0, 4 + cls, 0] = conf
    return out


def test_decode_thresholds_and_rescales():
    # letterbox identity: ratio 1, no pad, 640x640 original
    out = _fake_yolo_output(cx=320, cy=320, w=100, h=200, cls=3, conf=0.95)
    dets = decode_predictions(
        out, ratio=1.0, pad=(0, 0), orig_shape=(640, 640), conf_thresh=0.5, iou_thresh=0.5
    )
    assert isinstance(dets, Detections)
    assert len(dets) == 1
    x1, y1, x2, y2 = dets.boxes[0]
    assert np.isclose(x1, 270) and np.isclose(x2, 370)  # cx +/- w/2
    assert np.isclose(y1, 220) and np.isclose(y2, 420)  # cy +/- h/2
    assert dets.classes[0] == 3
    assert dets.scores[0] >= 0.5


def test_decode_filters_below_conf():
    out = _fake_yolo_output(cx=320, cy=320, w=100, h=200, cls=3, conf=0.2)
    dets = decode_predictions(
        out, ratio=1.0, pad=(0, 0), orig_shape=(640, 640), conf_thresh=0.5, iou_thresh=0.5
    )
    assert len(dets) == 0


def test_decode_undoes_letterbox_padding():
    # ratio 0.5 with 20px horizontal pad: original is larger than letterbox
    out = _fake_yolo_output(cx=120, cy=100, w=40, h=40, cls=0, conf=0.9)
    dets = decode_predictions(
        out, ratio=0.5, pad=(20, 0), orig_shape=(400, 400), conf_thresh=0.5, iou_thresh=0.5
    )
    assert len(dets) == 1
    x1, y1, x2, y2 = dets.boxes[0]
    # cx=120 -> (120-20)/0.5 = 200 ; w=40 -> 40/0.5=80 -> x in [160,240]
    assert np.isclose((x1 + x2) / 2, 200)
    assert np.isclose(x2 - x1, 80)
