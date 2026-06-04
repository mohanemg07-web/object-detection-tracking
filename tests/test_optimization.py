"""Unit tests for optimization helpers that don't need a model or GPU."""

import numpy as np
import pytest

from src.optimization.benchmark import BenchResult, format_markdown
from src.optimization.quantize_int8 import _list_images, letterbox, preprocess


@pytest.fixture
def bgr_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(120, 200, 3), dtype=np.uint8)


def test_letterbox_is_square_and_uint8(bgr_image):
    out = letterbox(bgr_image, new_shape=640)
    assert out.shape == (640, 640, 3)
    assert out.dtype == np.uint8


def test_letterbox_preserves_aspect_ratio(bgr_image):
    # 200x120 (w x h) -> scale by 640/200 = 3.2 -> content 640x384, padded vertically
    out = letterbox(bgr_image, new_shape=640, color=114)
    # top/bottom padding rows should equal the fill color
    assert (out[0] == 114).all()
    assert (out[-1] == 114).all()


def test_preprocess_shape_range_and_layout(bgr_image):
    out = preprocess(bgr_image, imgsz=640)
    assert out.shape == (1, 3, 640, 640)  # NCHW with batch
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_bgr_to_rgb_channel_swap():
    # solid blue image in BGR -> after RGB swap, the R channel (idx 0) is 0
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # blue in BGR
    out = preprocess(img, imgsz=32)
    # center pixel: R channel should be 0, B channel (idx 2) should be ~1
    assert out[0, 0, 16, 16] == pytest.approx(0.0)
    assert out[0, 2, 16, 16] == pytest.approx(1.0)


def test_list_images_even_stride_sampling(tmp_path):
    for i in range(20):
        (tmp_path / f"{i:03d}.jpg").write_bytes(b"x")
    sampled = _list_images(tmp_path, num_samples=5)
    assert len(sampled) == 5
    # all sampled files are distinct
    assert len({p.name for p in sampled}) == 5


def test_list_images_returns_all_when_fewer_than_requested(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.png").write_bytes(b"x")
    sampled = _list_images(tmp_path, num_samples=10)
    assert len(sampled) == 3


def test_list_images_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _list_images(tmp_path, num_samples=5)


def test_format_markdown_handles_missing_values():
    results = [
        BenchResult("PyTorch FP32", 142.0, 50.0, 20.0),
        BenchResult("ONNX INT8", 45.0, 30.0, 33.3),
        BenchResult("TensorRT INT8", None, None, None, "GPU only"),
    ]
    table = format_markdown(results)
    assert "| Model |" in table
    assert "PyTorch FP32" in table
    assert "142.0" in table
    assert "GPU only" in table
    # missing numbers render as em dash
    assert table.count("—") >= 3
