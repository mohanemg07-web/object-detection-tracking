"""Before/after visualization helpers for preprocessing transforms."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.transforms import Compose


def side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    labels: tuple[str, str] = ("before", "after"),
) -> np.ndarray:
    """Stack two BGR images horizontally with text labels.

    The images are resized to a common height (the smaller of the two) so
    the result is a single comparable strip.
    """
    h = min(before.shape[0], after.shape[0])

    def _fit(img: np.ndarray) -> np.ndarray:
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    left, right = _fit(before), _fit(after)
    canvas = np.hstack([left, right])

    for x, text in ((10, labels[0]), (left.shape[1] + 10, labels[1])):
        cv2.putText(canvas, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            canvas, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
        )
    return canvas


def visualize_pipeline(
    image: np.ndarray,
    pipeline: Compose,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Apply ``pipeline`` to ``image`` and return a before/after comparison.

    If ``save_path`` is given, the comparison image is written there.
    """
    processed = pipeline(image)
    comparison = side_by_side(image, processed)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), comparison)
    return comparison
