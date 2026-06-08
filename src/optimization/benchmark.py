"""Benchmark detector variants: size, latency, FPS (and optional mAP delta).

Measures whichever model files are present so it works incrementally:
  * PyTorch FP32   (weights/best.pt)        — needs ultralytics/torch
  * ONNX FP32      (weights/best.onnx)      — ONNX Runtime CPU
  * ONNX INT8      (weights/best.int8.onnx) — ONNX Runtime CPU
  * TensorRT INT8  (weights/best.int8.engine) — GPU only; skipped on CPU

Latency/FPS use warmup + repeated timed runs on a fixed-size random input
(or a sample image). Results print as a Markdown table that can be pasted
into the README; pass --write-readme to splice it in automatically.

Usage:
    python -m src.optimization.benchmark --weights-dir weights --runs 50
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BenchResult:
    name: str
    size_mb: float | None
    latency_ms: float | None
    fps: float | None
    note: str = ""


def _file_mb(path: Path) -> float | None:
    return path.stat().st_size / 1e6 if path.exists() else None


def _time_callable(fn, runs: int, warmup: int) -> tuple[float, float]:
    """Return (mean_latency_ms, fps) for a zero-arg callable."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean_s = statistics.mean(times)
    return mean_s * 1000.0, (1.0 / mean_s if mean_s > 0 else float("inf"))


def bench_onnx(path: Path, imgsz: int, runs: int, warmup: int, name: str) -> BenchResult:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((1, 3, imgsz, imgsz)).astype(np.float32)
    name_in = inp.name

    lat, fps = _time_callable(lambda: sess.run(None, {name_in: dummy}), runs, warmup)
    return BenchResult(name, _file_mb(path), lat, fps)


def bench_pytorch(path: Path, imgsz: int, runs: int, warmup: int) -> BenchResult:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(path)).model.float().eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)

    def _run():
        with torch.no_grad():
            model(dummy)

    lat, fps = _time_callable(_run, runs, warmup)
    return BenchResult("PyTorch FP32", _file_mb(path), lat, fps)


def bench_tensorrt(path: Path, imgsz: int, runs: int, warmup: int) -> BenchResult:
    """Time inference through a serialized TensorRT engine (TRT 10.x API).

    Uses the modern name-based tensor API (num_io_tensors / get_tensor_name /
    set_input_shape / set_tensor_address / execute_async_v3) — NOT the removed
    pre-10 binding API. Requires pycuda + a CUDA device. Raises on failure so
    the caller can fall back to a "present but not timed" note.
    """
    import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)
    import pycuda.driver as cuda
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError("failed to deserialize engine")
    context = engine.create_execution_context()

    stream = cuda.Stream()
    rng = np.random.default_rng(0)
    host_buffers: dict[str, np.ndarray] = {}
    device_buffers: dict[str, object] = {}

    # Discover I/O tensors by name (TRT 10 API).
    input_names: list[str] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        if is_input:
            # Pin the (possibly dynamic) input to the benchmark shape.
            context.set_input_shape(name, (1, 3, imgsz, imgsz))
            input_names.append(name)

    # Allocate host+device buffers for every I/O tensor at resolved shapes.
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(context.get_tensor_shape(name))
        host = rng.standard_normal(shape).astype(np.float32)
        host = np.ascontiguousarray(host)
        dev = cuda.mem_alloc(host.nbytes)
        host_buffers[name] = host
        device_buffers[name] = dev
        context.set_tensor_address(name, int(dev))

    # Upload inputs once (we time pure execution, like the ONNX path).
    for name in input_names:
        cuda.memcpy_htod_async(device_buffers[name], host_buffers[name], stream)
    stream.synchronize()

    def _run():
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    lat, fps = _time_callable(_run, runs, warmup)
    return BenchResult("TensorRT INT8", _file_mb(path), lat, fps)


def run_benchmarks(
    weights_dir: Path, imgsz: int, runs: int, warmup: int, stem: str = "best"
) -> list[BenchResult]:
    results: list[BenchResult] = []

    pt = weights_dir / f"{stem}.pt"
    if pt.exists():
        try:
            results.append(bench_pytorch(pt, imgsz, runs, warmup))
        except Exception as exc:  # noqa: BLE001
            results.append(BenchResult("PyTorch FP32", _file_mb(pt), None, None, f"skipped: {exc}"))
    else:
        results.append(BenchResult("PyTorch FP32", None, None, None, f"{pt.name} not found"))

    onnx_fp32 = weights_dir / f"{stem}.onnx"
    if onnx_fp32.exists():
        results.append(bench_onnx(onnx_fp32, imgsz, runs, warmup, "ONNX FP32"))
    else:
        results.append(BenchResult("ONNX FP32", None, None, None, f"{onnx_fp32.name} not found"))

    onnx_int8 = weights_dir / f"{stem}.int8.onnx"
    if onnx_int8.exists():
        results.append(bench_onnx(onnx_int8, imgsz, runs, warmup, "ONNX INT8"))
    else:
        results.append(BenchResult("ONNX INT8", None, None, None, f"{onnx_int8.name} not found"))

    engine = weights_dir / f"{stem}.int8.engine"
    if engine.exists():
        try:
            results.append(bench_tensorrt(engine, imgsz, runs, warmup))
        except Exception as exc:  # noqa: BLE001 - keep CPU rows even if TRT fails
            results.append(
                BenchResult(
                    "TensorRT INT8",
                    _file_mb(engine),
                    None,
                    None,
                    f"present but not timed ({exc})",
                )
            )
    else:
        results.append(
            BenchResult("TensorRT INT8", None, None, None, "GPU only — run on a CUDA machine")
        )

    return results


def format_markdown(results: list[BenchResult]) -> str:
    lines = [
        "| Model | Size (MB) | Latency (ms) | FPS | Note |",
        "|-------|:---------:|:------------:|:---:|------|",
    ]
    for r in results:
        size = f"{r.size_mb:.1f}" if r.size_mb is not None else "—"
        lat = f"{r.latency_ms:.1f}" if r.latency_ms is not None else "—"
        fps = f"{r.fps:.1f}" if r.fps is not None else "—"
        lines.append(f"| {r.name} | {size} | {lat} | {fps} | {r.note} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark detector variants (CPU).")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--stem", default="best", help="model filename stem (e.g. 'best')")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default=None, help="optional path to write the Markdown table")
    args = parser.parse_args(argv)

    weights_dir = Path(args.weights_dir)
    results = run_benchmarks(weights_dir, args.imgsz, args.runs, args.warmup, stem=args.stem)
    table = format_markdown(results)
    print("\n" + table + "\n")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"[bench] wrote table to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
