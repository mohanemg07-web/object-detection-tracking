"""Tests for the Gradio demo's pure logic (no gradio import needed)."""

import numpy as np

from app import demo_logic
from app.demo_logic import draw_detections, model_available, summarize
from src.inference.onnx_detector import Detections


def test_summarize_empty():
    out = summarize(np.array([], dtype=int), prefix="0 objects")
    assert "0 objects" in out
    assert "no detections" in out


def test_summarize_counts_by_class():
    # classes: two cars (3), one pedestrian (0)
    classes = np.array([3, 3, 0])
    out = summarize(classes, prefix="3 objects")
    assert "3 objects" in out
    assert "car: 2" in out
    assert "pedestrian: 1" in out


def test_summarize_handles_none():
    out = summarize(None, prefix="n/a")
    assert "no detections" in out


def test_model_available_false_for_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(demo_logic, "MODEL_PATH", str(tmp_path / "nope.onnx"))
    assert model_available() is False


def test_model_available_true_when_present(monkeypatch, tmp_path):
    fake = tmp_path / "best.int8.onnx"
    fake.write_bytes(b"not-a-real-model")
    monkeypatch.setattr(demo_logic, "MODEL_PATH", str(fake))
    assert model_available() is True


def test_detect_image_none_input_returns_prompt():
    out, msg = demo_logic.detect_image(None)
    assert out is None
    assert "Upload an image" in msg


def test_detect_image_missing_model_returns_message(monkeypatch, tmp_path):
    monkeypatch.setattr(demo_logic, "MODEL_PATH", str(tmp_path / "nope.onnx"))
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    out, msg = demo_logic.detect_image(img)
    # returns the original image plus a "model not found" message
    assert out is img
    assert "Model not found" in msg


def test_detect_video_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(demo_logic, "MODEL_PATH", str(tmp_path / "nope.onnx"))
    out, msg = demo_logic.detect_video("some_video.mp4")
    assert out is None
    assert "Model not found" in msg


def test_draw_detections_runs_and_returns_same_shape():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = Detections(
        boxes=np.array([[10.0, 10.0, 50.0, 60.0]]),
        scores=np.array([0.9]),
        classes=np.array([3]),
    )
    out = draw_detections(frame.copy(), dets)
    assert out.shape == frame.shape
    # something was drawn (frame no longer all zeros)
    assert out.sum() > 0
