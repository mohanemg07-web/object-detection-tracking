"""Static INT8 quantization of the ONNX detector for the CPU demo path.

Uses ONNX Runtime's static quantization with a calibration data reader
that streams a small subset of real images through the model's exact
preprocessing (letterbox to imgsz, BGR->RGB, /255, CHW). Static (vs
dynamic) quantization is what yields the size/latency win for conv-heavy
detectors, at the cost of needing calibration images.

Runs on CPU. Produces ``<model>.int8.onnx``.

Usage:
    python -m src.optimization.quantize_int8 \
        --model weights/best.onnx \
        --calib-dir data/yolo/VisDrone-DET/images/val \
        --num-samples 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def letterbox(image: np.ndarray, new_shape: int = 640, color: int = 114) -> np.ndarray:
    """Resize+pad a BGR image to a square ``new_shape`` keeping aspect ratio.

    Mirrors Ultralytics' letterbox so calibration statistics match what the
    model sees at inference time.
    """
    import cv2

    h, w = image.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def preprocess(image: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """BGR uint8 HxWx3 -> NCHW float32 [0,1] RGB, letterboxed to imgsz."""
    lb = letterbox(image, imgsz)
    rgb = lb[:, :, ::-1]  # BGR -> RGB
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return chw[None, ...]  # add batch dim


def _list_images(calib_dir: Path, num_samples: int) -> list[Path]:
    files = [p for p in sorted(calib_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"no calibration images in {calib_dir}")
    if num_samples and len(files) > num_samples:
        # Even stride sampling for a representative subset.
        step = len(files) / num_samples
        files = [files[int(i * step)] for i in range(num_samples)]
    return files


def build_calibration_reader(input_name: str, calib_dir: Path, num_samples: int, imgsz: int):
    """Create an ONNX Runtime CalibrationDataReader over calib images."""
    import cv2
    from onnxruntime.quantization import CalibrationDataReader

    files = _list_images(calib_dir, num_samples)
    print(f"[quant] calibrating on {len(files)} images from {calib_dir}")

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._it = iter(files)

        def get_next(self):
            for path in self._it:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                return {input_name: preprocess(img, imgsz)}
            return None

        def rewind(self) -> None:
            self._it = iter(files)

    return _Reader()


def quantize(
    model_path: str | Path,
    calib_dir: str | Path,
    out_path: str | Path | None = None,
    num_samples: int = 200,
    imgsz: int = 640,
    per_channel: bool = True,
) -> Path:
    """Statically quantize an ONNX model to INT8 and return the output path."""
    import onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.preprocess import quant_pre_process

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path} (run export_onnx first)")
    calib_dir = Path(calib_dir)
    out_path = Path(out_path) if out_path else model_path.with_suffix(".int8.onnx")

    # Pre-process (shape inference + folding) improves static-quant quality.
    prepped = model_path.with_suffix(".prep.onnx")
    quant_pre_process(str(model_path), str(prepped))

    model = onnx.load(str(prepped))
    input_name = model.graph.input[0].name

    reader = build_calibration_reader(input_name, calib_dir, num_samples, imgsz)
    quantize_static(
        str(prepped),
        str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    prepped.unlink(missing_ok=True)

    fp32_mb = model_path.stat().st_size / 1e6
    int8_mb = out_path.stat().st_size / 1e6
    reduction = 100.0 * (1 - int8_mb / fp32_mb) if fp32_mb else 0.0
    print(
        f"[quant] {model_path.name}: {fp32_mb:.1f} MB -> {out_path.name}: {int8_mb:.1f} MB "
        f"({reduction:.1f}% smaller)"
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static INT8 quantization of an ONNX detector.")
    parser.add_argument("--model", default="weights/best.onnx")
    parser.add_argument("--calib-dir", default="data/yolo/VisDrone-DET/images/val")
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-per-channel", action="store_true")
    args = parser.parse_args(argv)

    quantize(
        args.model,
        args.calib_dir,
        out_path=args.out,
        num_samples=args.num_samples,
        imgsz=args.imgsz,
        per_channel=not args.no_per_channel,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
