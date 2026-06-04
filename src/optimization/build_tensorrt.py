"""Build a TensorRT INT8 engine from the ONNX detector.  *** GPU ONLY ***

This requires an NVIDIA GPU + CUDA + TensorRT + pycuda. It will refuse to
run otherwise. Use the Colab T4 notebook (notebooks/tensorrt_int8.ipynb)
or a local CUDA box. The CPU demo never touches this — it uses ONNX-INT8.

Pipeline:
    ONNX -> TensorRT builder with an Int8EntropyCalibrator2 fed by a
    calibration image subset -> serialized .engine on disk.

Usage (on a GPU machine):
    python -m src.optimization.build_tensorrt \
        --onnx weights/best.onnx \
        --calib-dir data/yolo/VisDrone-DET/images/val \
        --engine weights/best.int8.engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.optimization.quantize_int8 import _list_images, preprocess


def _require_gpu_stack() -> tuple:
    """Import the GPU-only stack or exit with a clear message."""
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "TensorRT is not installed. This step is GPU-only.\n"
            "Run it on a CUDA machine / Colab T4 with:\n"
            "  pip install -r requirements-gpu.txt\n"
            "The CPU demo does NOT need this; it uses the ONNX-INT8 model."
        ) from exc
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError as exc:
        raise SystemExit("pycuda not installed (GPU-only). See requirements-gpu.txt.") from exc

    import tensorrt as trt

    return trt, cuda


def _make_calibrator(trt, cuda, calib_dir: Path, num_samples: int, imgsz: int, cache: Path):
    """Build an INT8 entropy calibrator streaming calibration images."""

    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.files = _list_images(calib_dir, num_samples)
            self.idx = 0
            self.imgsz = imgsz
            self.cache_file = cache
            self.device_input = cuda.mem_alloc(int(np.prod((1, 3, imgsz, imgsz)) * 4))

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names):  # noqa: ARG002
            import cv2

            if self.idx >= len(self.files):
                return None
            img = cv2.imread(str(self.files[self.idx]))
            self.idx += 1
            if img is None:
                return self.get_batch(names)
            batch = np.ascontiguousarray(preprocess(img, self.imgsz))
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_file.exists():
                return self.cache_file.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self.cache_file.write_bytes(cache)

    return _EntropyCalibrator()


def build_engine(
    onnx_path: str | Path,
    calib_dir: str | Path,
    engine_path: str | Path,
    num_samples: int = 200,
    imgsz: int = 640,
    workspace_gb: int = 4,
) -> Path:
    """Build + serialize an INT8 TensorRT engine. GPU required."""
    trt, cuda = _require_gpu_stack()

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    cache = engine_path.with_suffix(".calib.cache")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path} (run export_onnx first)")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse ONNX:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if not builder.platform_has_fast_int8:
        print("[trt] WARNING: platform reports no fast INT8; engine may fall back.")
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = _make_calibrator(trt, cuda, Path(calib_dir), num_samples, imgsz, cache)

    print("[trt] building INT8 engine (this can take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed (returned None)")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"[trt] engine written: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    return engine_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a TensorRT INT8 engine (GPU only).")
    parser.add_argument("--onnx", default="weights/best.onnx")
    parser.add_argument("--calib-dir", default="data/yolo/VisDrone-DET/images/val")
    parser.add_argument("--engine", default="weights/best.int8.engine")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace-gb", type=int, default=4)
    args = parser.parse_args(argv)

    build_engine(
        args.onnx,
        args.calib_dir,
        args.engine,
        num_samples=args.num_samples,
        imgsz=args.imgsz,
        workspace_gb=args.workspace_gb,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
