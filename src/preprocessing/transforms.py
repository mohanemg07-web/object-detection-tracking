"""Composable, config-driven OpenCV preprocessing transforms.

Each transform is a small callable that takes a BGR ``np.ndarray`` (the
OpenCV convention) and returns a transformed BGR image of the same dtype.
Transforms are pure (no global state) and individually testable. Compose
them with :class:`Compose` or build a pipeline from config with
:func:`build_pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np


class Transform(Protocol):
    """A preprocessing transform: BGR image in, BGR image out."""

    def __call__(self, image: np.ndarray) -> np.ndarray: ...


def _check_bgr(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected np.ndarray, got {type(image)!r}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape {image.shape}")


@dataclass(frozen=True)
class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization.

    Applied on the L channel of LAB by default so contrast is enhanced
    without shifting hue/saturation. Set ``color_space='gray'`` to operate
    on a grayscale copy broadcast back to 3 channels.
    """

    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)
    color_space: str = "lab"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        _check_bgr(image)
        clahe = cv2.createCLAHE(
            clipLimit=float(self.clip_limit),
            tileGridSize=(int(self.tile_grid_size[0]), int(self.tile_grid_size[1])),
        )
        if self.color_space == "lab":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            l_eq = clahe.apply(l_chan)
            merged = cv2.merge((l_eq, a_chan, b_chan))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        if self.color_space == "gray":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            eq = clahe.apply(gray)
            return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"unsupported color_space {self.color_space!r} (use 'lab' or 'gray')")


@dataclass(frozen=True)
class PerspectiveCorrection:
    """Warp a source quadrilateral to a destination rectangle.

    Points are given as fractions of (width, height) so a config works at
    any resolution. ``output_size`` is ``(w, h)`` in pixels; when ``None``
    the input size is preserved.
    """

    src_points: tuple[tuple[float, float], ...]
    dst_points: tuple[tuple[float, float], ...]
    output_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if len(self.src_points) != 4 or len(self.dst_points) != 4:
            raise ValueError("perspective correction needs exactly 4 src and 4 dst points")

    @staticmethod
    def _to_pixels(points, w: int, h: int) -> np.ndarray:
        return np.array([[px * w, py * h] for px, py in points], dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        _check_bgr(image)
        h, w = image.shape[:2]
        out_w, out_h = self.output_size if self.output_size else (w, h)
        src = self._to_pixels(self.src_points, w, h)
        dst = self._to_pixels(self.dst_points, out_w, out_h)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (int(out_w), int(out_h)))


@dataclass
class Compose:
    """Run a list of transforms left-to-right."""

    transforms: list[Transform] = field(default_factory=list)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image

    def __len__(self) -> int:
        return len(self.transforms)


def build_pipeline(config: dict) -> Compose:
    """Build a :class:`Compose` pipeline from a preprocess config dict.

    Only transforms with ``enabled: true`` are included, in a fixed order
    (CLAHE then perspective). Unknown keys are ignored so the config can
    carry extra metadata.
    """
    transforms: list[Transform] = []

    clahe_cfg = config.get("clahe", {})
    if clahe_cfg.get("enabled", False):
        grid = clahe_cfg.get("tile_grid_size", [8, 8])
        transforms.append(
            CLAHE(
                clip_limit=clahe_cfg.get("clip_limit", 2.0),
                tile_grid_size=(int(grid[0]), int(grid[1])),
                color_space=clahe_cfg.get("color_space", "lab"),
            )
        )

    persp_cfg = config.get("perspective", {})
    if persp_cfg.get("enabled", False):
        out = persp_cfg.get("output_size")
        transforms.append(
            PerspectiveCorrection(
                src_points=tuple(tuple(p) for p in persp_cfg["src_points"]),
                dst_points=tuple(tuple(p) for p in persp_cfg["dst_points"]),
                output_size=tuple(out) if out else None,
            )
        )

    return Compose(transforms)
