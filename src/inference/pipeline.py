"""Unified preprocess -> detect -> track pipeline with annotation + FPS.

Backend-agnostic: pair any detector exposing ``detect(bgr) -> Detections``
(see :class:`OnnxDetector`) with :class:`ByteTracker`. Optionally runs the
OpenCV preprocessing pipeline first. Draws boxes + track ids + class names
and an FPS overlay, and can consume a video file or webcam index.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.data.visdrone import CLASS_NAMES
from src.preprocessing.transforms import Compose
from src.tracking.bytetrack import ByteTracker


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic distinct BGR color per track id."""
    rng = np.random.default_rng(track_id * 9973 + 1)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


@dataclass
class FrameResult:
    frame: np.ndarray  # annotated BGR frame
    num_tracks: int
    fps: float


class TrackingPipeline:
    """Compose preprocessing, a detector, and ByteTrack into one callable."""

    def __init__(
        self,
        detector,
        tracker: ByteTracker | None = None,
        preprocess: Compose | None = None,
        class_names: list[str] | None = None,
        fps_window: int = 30,
    ) -> None:
        self.detector = detector
        self.tracker = tracker or ByteTracker()
        self.preprocess = preprocess
        self.class_names = class_names or CLASS_NAMES
        self._times: deque[float] = deque(maxlen=fps_window)

    def _class_name(self, cls: int) -> str:
        return self.class_names[cls] if 0 <= cls < len(self.class_names) else str(cls)

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Run one frame end-to-end and return the annotated result."""
        t0 = time.perf_counter()

        work = self.preprocess(frame) if self.preprocess else frame
        dets = self.detector.detect(work)
        tracks = self.tracker.update(dets.boxes, dets.scores, dets.classes)

        annotated = self.annotate(frame.copy(), tracks)

        dt = time.perf_counter() - t0
        self._times.append(dt)
        fps = len(self._times) / sum(self._times) if self._times else 0.0

        self._draw_fps(annotated, fps)
        return FrameResult(annotated, len(tracks), fps)

    def annotate(self, frame: np.ndarray, tracks) -> np.ndarray:
        import cv2

        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.xyxy)
            color = _color_for_id(t.track_id)
            label = f"#{t.track_id} {self._class_name(t.cls)} {t.score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _draw_fps(frame: np.ndarray, fps: float) -> None:
        import cv2

        text = f"FPS: {fps:4.1f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
        )
