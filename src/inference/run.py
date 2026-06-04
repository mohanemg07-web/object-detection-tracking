"""Run the unified detect+track pipeline on a video file or webcam.

Uses the ONNX detector (CPU by default) + ByteTrack, optionally with the
OpenCV preprocessing pipeline. Writes an annotated video and/or shows a
live window.

Usage:
    # webcam 0, live window
    python -m src.inference.run --model weights/best.int8.onnx --source 0

    # video file -> annotated mp4
    python -m src.inference.run --model weights/best.int8.onnx \
        --source clip.mp4 --save outputs/clip_tracked.mp4 --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.inference.onnx_detector import OnnxDetector
from src.inference.pipeline import TrackingPipeline
from src.preprocessing.transforms import build_pipeline
from src.tracking.bytetrack import ByteTracker


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _open_source(source: str):
    import cv2

    # numeric -> webcam index
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source: {source}")
    return cap


def main(argv: list[str] | None = None) -> int:
    import cv2

    parser = argparse.ArgumentParser(description="Detect + track on video/webcam.")
    parser.add_argument("--model", default="weights/best.int8.onnx")
    parser.add_argument("--source", default="0", help="video path or webcam index")
    parser.add_argument("--tracker-config", default="configs/bytetrack.yaml")
    parser.add_argument("--preprocess-config", default=None, help="enable OpenCV preprocessing")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", default=None, help="output annotated video path")
    parser.add_argument("--no-show", action="store_true", help="do not open a display window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=all)")
    args = parser.parse_args(argv)

    detector = OnnxDetector(
        args.model, imgsz=args.imgsz, conf_thresh=args.conf, iou_thresh=args.iou
    )
    tracker = ByteTracker(_load_yaml(args.tracker_config))
    preprocess = (
        build_pipeline(_load_yaml(args.preprocess_config)) if args.preprocess_config else None
    )
    pipeline = TrackingPipeline(detector, tracker, preprocess=preprocess)

    cap = _open_source(args.source)
    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process_frame(frame)
            frames += 1

            if writer is not None:
                writer.write(result.frame)
            if not args.no_show:
                cv2.imshow("detect+track", result.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print(f"processed {frames} frames")
    if args.save:
        print(f"saved annotated video -> {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
