"""Evaluate tracking with MOTA/MOTP on a VisDrone-MOT sequence.

VisDrone-MOT ground-truth lines (per object, per frame):

    <frame>,<target_id>,<bbox_left>,<bbox_top>,<bbox_w>,<bbox_h>,
    <score>,<category>,<truncation>,<occlusion>

We run the detector+ByteTracker frame-by-frame over a sequence's images,
accumulate matches against ground truth with ``motmetrics`` (IoU-based,
distance threshold 0.5), and report MOTA, MOTP, IDF1, ID switches, etc.

Usage:
    python -m src.tracking.evaluate_mota \
        --model weights/best.int8.onnx \
        --seq-images data/raw/VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
        --gt data/raw/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.tracking.visdrone_mot import load_mot_gt

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _iter_frames(seq_dir: Path):
    """Yield (frame_index_1based, image_path) for a sequence directory."""
    files = [p for p in sorted(seq_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    yield from enumerate(files, start=1)


def evaluate_sequence(
    detector,
    tracker_config: dict,
    seq_dir: Path,
    gt_by_frame: dict[int, list[tuple]],
    distance_threshold: float = 0.5,
) -> dict:
    """Run tracking over a sequence and accumulate MOT metrics.

    ``gt_by_frame`` maps frame -> list of (target_id, x1, y1, x2, y2).
    Returns a dict of summary metrics.
    """
    import cv2
    import motmetrics as mm

    from src.tracking.bytetrack import ByteTracker

    tracker = ByteTracker(tracker_config)
    acc = mm.MOTAccumulator(auto_id=False)

    for frame_idx, img_path in _iter_frames(seq_dir):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        dets = detector.detect(image)
        tracks = tracker.update(dets.boxes, dets.scores, dets.classes)

        gt_objs = gt_by_frame.get(frame_idx, [])
        gt_ids = [o[0] for o in gt_objs]
        gt_boxes = np.array([o[1:5] for o in gt_objs], dtype=float) if gt_objs else np.empty((0, 4))

        hyp_ids = [t.track_id for t in tracks]
        hyp_boxes = np.array([t.xyxy for t in tracks], dtype=float) if tracks else np.empty((0, 4))

        # motmetrics expects (x, y, w, h) and a distance matrix (1 - IoU).
        gt_xywh = _to_xywh(gt_boxes)
        hyp_xywh = _to_xywh(hyp_boxes)
        dist = mm.distances.iou_matrix(gt_xywh, hyp_xywh, max_iou=distance_threshold)
        acc.update(gt_ids, hyp_ids, dist, frameid=frame_idx)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota",
            "motp",
            "idf1",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "num_objects",
            "mostly_tracked",
            "mostly_lost",
        ],
        name="seq",
    )
    return summary.to_dict("records")[0]


def _to_xywh(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0, 4))
    out = boxes.copy().astype(float)
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description="MOTA/MOTP evaluation (VisDrone-MOT).")
    parser.add_argument("--model", default="weights/best.int8.onnx")
    parser.add_argument("--seq-images", required=True, help="sequence image directory")
    parser.add_argument("--gt", required=True, help="VisDrone-MOT gt .txt for the sequence")
    parser.add_argument("--tracker-config", default="configs/bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    from src.inference.onnx_detector import OnnxDetector

    detector = OnnxDetector(args.model, imgsz=args.imgsz, conf_thresh=args.conf)
    with Path(args.tracker_config).open("r", encoding="utf-8") as fh:
        tracker_config = yaml.safe_load(fh) or {}

    gt_by_frame = load_mot_gt(Path(args.gt))
    metrics = evaluate_sequence(detector, tracker_config, Path(args.seq_images), gt_by_frame)

    print("\n=== MOT metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:22s}: {v:.4f}")
        else:
            print(f"  {k:22s}: {v}")
    print(f"\nMOTA={metrics['mota']:.4f}  MOTP={metrics['motp']:.4f}  IDF1={metrics['idf1']:.4f}")
    return 0


def aggregate_metrics(per_seq: list[dict]) -> dict:
    """Average key metrics across multiple sequences (simple mean)."""
    if not per_seq:
        return {}
    keys = ["mota", "motp", "idf1"]
    agg: dict[str, float] = defaultdict(float)
    for m in per_seq:
        for k in keys:
            agg[k] += float(m.get(k, 0.0))
    return {k: agg[k] / len(per_seq) for k in keys}


if __name__ == "__main__":
    sys.exit(main())
