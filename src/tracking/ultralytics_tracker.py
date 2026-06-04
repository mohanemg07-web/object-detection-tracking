"""Thin wrapper around Ultralytics' built-in ByteTrack/BoT-SORT tracker.

This is the GPU-friendly path that uses the fine-tuned ``best.pt`` directly
with Ultralytics' ``model.track(...)`` and the tracker YAML configs
(configs/bytetrack.yaml or configs/botsort.yaml). The CPU demo uses the
self-contained ONNX + ByteTracker path instead; this exists so the same
tracker config drives both, and for high-FPS GPU tracking.

Usage:
    python -m src.tracking.ultralytics_tracker \
        --weights weights/best.pt --source clip.mp4 \
        --tracker configs/bytetrack.yaml --save outputs/tracked.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def track(
    weights: str | Path,
    source: str,
    tracker: str = "configs/bytetrack.yaml",
    conf: float = 0.25,
    iou: float = 0.45,
    save: bool = True,
    show: bool = False,
    device: str | int | None = None,
):
    """Run Ultralytics tracking; yields per-frame Results (also persisted)."""
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"weights not found: {weights} (train first, then place best.pt here)"
        )

    model = YOLO(str(weights))
    # stream=True keeps memory flat over long videos.
    return model.track(
        source=source,
        tracker=str(tracker),
        conf=conf,
        iou=iou,
        persist=True,
        stream=True,
        save=save,
        show=show,
        device=device,
        verbose=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ultralytics ByteTrack/BoT-SORT tracking.")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", required=True, help="video path or webcam index")
    parser.add_argument("--tracker", default="configs/bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None, help="cuda index or 'cpu'")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    n_frames = 0
    total_tracks = 0
    for result in track(
        args.weights,
        args.source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        save=not args.no_save,
        show=args.show,
        device=args.device,
    ):
        n_frames += 1
        if result.boxes is not None and result.boxes.id is not None:
            total_tracks += len(result.boxes.id)

    avg = total_tracks / n_frames if n_frames else 0.0
    print(f"processed {n_frames} frames; avg {avg:.1f} tracked objects/frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
