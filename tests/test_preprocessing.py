"""Unit tests for the OpenCV preprocessing transforms."""

import numpy as np
import pytest

from src.preprocessing.transforms import (
    CLAHE,
    Compose,
    PerspectiveCorrection,
    build_pipeline,
)


@pytest.fixture
def gradient_image():
    """A low-contrast-ish BGR gradient image for CLAHE to act on."""
    h, w = 64, 96
    base = np.tile(np.linspace(40, 120, w, dtype=np.uint8), (h, 1))
    return np.stack([base, base, base], axis=-1)


def test_clahe_preserves_shape_and_dtype(gradient_image):
    out = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))(gradient_image)
    assert out.shape == gradient_image.shape
    assert out.dtype == np.uint8


def test_clahe_increases_contrast(gradient_image):
    out = CLAHE(clip_limit=4.0, tile_grid_size=(8, 8))(gradient_image)
    # CLAHE should expand the dynamic range (std typically increases)
    assert out.std() >= gradient_image.std()


def test_clahe_gray_mode(gradient_image):
    out = CLAHE(color_space="gray")(gradient_image)
    assert out.shape == gradient_image.shape
    # gray mode broadcasts one channel -> all 3 channels equal
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 1], out[:, :, 2])


def test_clahe_rejects_bad_color_space(gradient_image):
    with pytest.raises(ValueError):
        CLAHE(color_space="hsv")(gradient_image)


def test_clahe_rejects_non_bgr():
    with pytest.raises(ValueError):
        CLAHE()(np.zeros((10, 10), dtype=np.uint8))


def test_identity_perspective_returns_equivalent_image(gradient_image):
    # mapping the full frame to itself is (near) identity
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    out = PerspectiveCorrection(src_points=corners, dst_points=corners)(gradient_image)
    assert out.shape == gradient_image.shape
    # allow tiny interpolation differences at edges
    assert np.abs(out.astype(int) - gradient_image.astype(int)).mean() < 1.0


def test_perspective_custom_output_size(gradient_image):
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    out = PerspectiveCorrection(src_points=corners, dst_points=corners, output_size=(48, 32))(
        gradient_image
    )
    assert out.shape[:2] == (32, 48)


def test_perspective_requires_four_points():
    with pytest.raises(ValueError):
        PerspectiveCorrection(src_points=((0, 0),), dst_points=((0, 0),))


def test_compose_applies_in_order(gradient_image):
    pipeline = Compose([CLAHE(), CLAHE(color_space="gray")])
    out = pipeline(gradient_image)
    assert out.shape == gradient_image.shape
    assert len(pipeline) == 2


def test_build_pipeline_from_config(gradient_image):
    config = {
        "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
        "perspective": {"enabled": False},
    }
    pipeline = build_pipeline(config)
    assert len(pipeline) == 1
    out = pipeline(gradient_image)
    assert out.shape == gradient_image.shape


def test_build_pipeline_all_disabled_is_identity(gradient_image):
    config = {"clahe": {"enabled": False}, "perspective": {"enabled": False}}
    pipeline = build_pipeline(config)
    assert len(pipeline) == 0
    assert np.array_equal(pipeline(gradient_image), gradient_image)


def test_build_pipeline_with_perspective(gradient_image):
    config = {
        "clahe": {"enabled": True},
        "perspective": {
            "enabled": True,
            "src_points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "dst_points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "output_size": None,
        },
    }
    pipeline = build_pipeline(config)
    assert len(pipeline) == 2
    out = pipeline(gradient_image)
    assert out.shape == gradient_image.shape
