"""Pure inference logic for the Gradio demo (no gradio import).

Kept separate from ``app.py`` so it is unit-testable without the gradio
runtime and so the UI layer stays thin. Everything here runs on CPU via
ONNX Runtime + the dependency-light ByteTracker.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the repo importable whether launched from root or from app/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.visdrone import CLASS_NAMES  # noqa: E402
from src.inference.onnx_detector import OnnxDetector  # noqa: E402
from src.inference.pipeline import TrackingPipeline  # noqa: E402
from src.tracking.bytetrack import ByteTracker  # noqa: E402

# Config via env (Spaces vars) with sensible defaults.
MODEL_PATH = os.environ.get("MODEL_PATH", str(REPO_ROOT / "weights" / "best.int8.onnx"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
CONF = float(os.environ.get("CONF", "0.25"))
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "300"))

_DETECTOR: OnnxDetector | None = None


def model_available() -> bool:
    return Path(MODEL_PATH).exists()


def get_detector() -> OnnxDetector:
    """Lazily construct the ONNX detector (loaded once, reused)."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = OnnxDetector(
            MODEL_PATH, imgsz=IMGSZ, conf_thresh=CONF, providers=["CPUExecutionProvider"]
        )
    return _DETECTOR


def missing_model_message() -> str:
    return (
        f"Model not found at '{MODEL_PATH}'.\n"
        "Add the ONNX INT8 weights (best.int8.onnx) to weights/ or set the "
        "MODEL_PATH env var. On Spaces, upload the model or fetch it at "
        "startup. The demo uses ONNX Runtime on CPU (no GPU/TensorRT)."
    )


def summarize(classes: np.ndarray | None, prefix: str) -> str:
    """Human-readable per-class count summary for the info textbox."""
    if classes is None or len(classes) == 0:
        return f"{prefix}\n(no detections above threshold)"
    counts: dict[str, int] = {}
    for c in classes:
        name = CLASS_NAMES[int(c)] if 0 <= int(c) < len(CLASS_NAMES) else str(int(c))
        counts[name] = counts.get(name, 0) + 1
    breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    return f"{prefix}\n{breakdown}"


def draw_detections(frame_bgr: np.ndarray, dets) -> np.ndarray:
    """Draw detection boxes + class labels on a BGR frame."""
    import cv2

    for box, score, cls in zip(dets.boxes, dets.scores, dets.classes, strict=False):
        x1, y1, x2, y2 = (int(v) for v in box)
        name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
        label = f"{name} {score:.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 200, 0), -1)
        cv2.putText(
            frame_bgr,
            label,
            (x1 + 1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame_bgr


def detect_image(image_rgb: np.ndarray):
    """Single-image detection. Gradio passes/returns RGB arrays."""
    if image_rgb is None:
        return None, "Upload an image first."
    if not model_available():
        return image_rgb, missing_model_message()

    import cv2

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    dets = get_detector().detect(bgr)
    annotated = draw_detections(bgr.copy(), dets)
    dt = time.perf_counter() - t0

    summary = summarize(dets.classes, prefix=f"{len(dets)} objects in {dt * 1000:.0f} ms")
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), summary


def detect_video(video_path: str):
    """Detection + tracking over a video; returns annotated mp4 path + summary."""
    if not video_path:
        return None, "Upload a video first."
    if not model_available():
        return None, missing_model_message()

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not read the uploaded video."

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = str(Path(video_path).with_name("tracked_output.mp4"))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    pipeline = TrackingPipeline(get_detector(), ByteTracker(), class_names=CLASS_NAMES)
    fps_samples: list[float] = []
    frames = 0
    try:
        while frames < MAX_VIDEO_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process_frame(frame)
            writer.write(result.frame)
            fps_samples.append(result.fps)
            frames += 1
    finally:
        cap.release()
        writer.release()

    avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
    note = "" if frames < MAX_VIDEO_FRAMES else f" (capped at {MAX_VIDEO_FRAMES} frames)"
    return out_path, f"Processed {frames} frames{note} · avg {avg_fps:.1f} FPS (CPU)"
