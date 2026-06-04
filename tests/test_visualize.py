"""Tests for the preprocessing visualization helper."""

import cv2
import numpy as np

from src.preprocessing.transforms import CLAHE, Compose
from src.preprocessing.visualize import side_by_side, visualize_pipeline


def _image(h=50, w=70):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_side_by_side_width_is_sum():
    a = _image(50, 70)
    b = _image(50, 40)
    out = side_by_side(a, b)
    assert out.shape[0] == 50  # common height
    assert out.shape[1] == 70 + 40  # widths concatenated
    assert out.shape[2] == 3


def test_side_by_side_resizes_to_common_height():
    a = _image(80, 60)
    b = _image(40, 60)  # smaller height -> target height 40
    out = side_by_side(a, b)
    assert out.shape[0] == 40


def test_visualize_pipeline_writes_file(tmp_path):
    img = _image()
    pipeline = Compose([CLAHE()])
    out_path = tmp_path / "viz" / "compare.jpg"
    comparison = visualize_pipeline(img, pipeline, save_path=out_path)
    assert out_path.exists()
    written = cv2.imread(str(out_path))
    assert written is not None
    assert written.shape == comparison.shape
