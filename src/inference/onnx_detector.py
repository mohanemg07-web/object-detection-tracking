"""ONNX Runtime detector for YOLOv8 (CPU-friendly, used by the demo).

Loads a YOLOv8 ONNX graph (FP32 or static-INT8) and runs the full
detection path: letterbox preprocess -> forward -> decode -> per-class
NMS -> rescale boxes back to the original image. No torch dependency, so
the demo stays light (onnxruntime + numpy + opencv).

YOLOv8 export output is ``(1, 4 + num_classes, num_anchors)`` with boxes
as ``(cx, cy, w, h)`` in letterboxed-pixel space and post-sigmoid class
scores. We transpose, threshold, NMS, then undo the letterbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detections:
    """Detections in original-image pixel coords.

    ``boxes`` is ``(N, 4)`` as ``(x1, y1, x2, y2)``; ``scores`` is ``(N,)``;
    ``classes`` is ``(N,)`` int class ids.
    """

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


def letterbox(image: np.ndarray, new_shape: int = 640, color: int = 114):
    """Resize+pad BGR image to square; return (canvas, ratio, (dw, dh))."""
    import cv2

    h, w = image.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw, dh = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[dh : dh + nh, dw : dw + nw] = resized
    return canvas, r, (dw, dh)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Pure-numpy non-max suppression. Returns kept indices (score order)."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_thresh]
    return keep


def decode_predictions(
    output: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    conf_thresh: float,
    iou_thresh: float,
) -> Detections:
    """Decode raw YOLOv8 ONNX output into rescaled, NMS'd detections."""
    # output: (1, 4+nc, anchors) -> (anchors, 4+nc)
    pred = np.squeeze(output, axis=0).T
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    mask = confidences >= conf_thresh
    if not np.any(mask):
        return Detections(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int))
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # xywh (letterbox space) -> xyxy
    cx, cy, w, h = boxes_xywh.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # undo letterbox: subtract pad, divide by ratio
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= ratio

    oh, ow = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

    # per-class NMS
    keep_all: list[int] = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        kept = nms(boxes[idx], confidences[idx], iou_thresh)
        keep_all.extend(idx[kept].tolist())

    keep_all = sorted(keep_all, key=lambda i: -confidences[i])
    return Detections(boxes[keep_all], confidences[keep_all], class_ids[keep_all].astype(int))


class OnnxDetector:
    """YOLOv8 detector backed by ONNX Runtime (CPU by default)."""

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, image: np.ndarray) -> Detections:
        """Run detection on a BGR image; return original-coord detections."""
        canvas, ratio, pad = letterbox(image, self.imgsz)
        rgb = canvas[:, :, ::-1]
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        output = self.session.run(None, {self.input_name: blob})[0]
        return decode_predictions(
            output, ratio, pad, image.shape[:2], self.conf_thresh, self.iou_thresh
        )
