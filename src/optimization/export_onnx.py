"""Export a fine-tuned YOLOv8 model to ONNX and verify PyTorch<->ORT parity.

Runs on CPU. The exported graph uses a pinned opset and dynamic batch axis
so the same file serves both the CPU demo and (after INT8) benchmarking.

Usage:
    python -m src.optimization.export_onnx --weights weights/best.pt
    python -m src.optimization.export_onnx --weights weights/best.pt \
        --imgsz 640 --opset 12 --no-simplify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_OPSET = 13


def export(
    weights: str | Path,
    imgsz: int = 640,
    opset: int = DEFAULT_OPSET,
    dynamic: bool = True,
    simplify: bool = True,
    half: bool = False,
) -> Path:
    """Export ``weights`` to ONNX, returning the .onnx path.

    Uses Ultralytics' exporter so the ONNX graph matches the model's
    expected pre/post-processing. Dynamic axes keep batch size flexible.
    """
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"weights not found: {weights}\n"
            "Train first (notebooks/train_colab.ipynb) and place best.pt there."
        )

    model = YOLO(str(weights))
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        half=half,
    )
    out_path = Path(out)
    print(f"[export] ONNX written: {out_path}")
    return out_path


def verify_parity(
    weights: str | Path,
    onnx_path: str | Path,
    imgsz: int = 640,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> bool:
    """Compare PyTorch vs ONNX Runtime outputs on a random input.

    Returns True if the raw model outputs match within tolerance. This
    checks the network numerics (pre-NMS), which is the right place to
    catch export drift. Detection-level mAP parity is covered separately
    by the benchmark script on real data.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    torch_model = model.model.float().eval()

    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((1, 3, imgsz, imgsz)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(dummy))
    torch_arr = torch_out[0] if isinstance(torch_out, (list, tuple)) else torch_out
    torch_arr = torch_arr.cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]

    if torch_arr.shape != ort_out.shape:
        print(f"[parity] shape mismatch: torch {torch_arr.shape} vs ort {ort_out.shape}")
        return False

    ok = np.allclose(torch_arr, ort_out, rtol=rtol, atol=atol)
    max_diff = float(np.max(np.abs(torch_arr - ort_out)))
    print(f"[parity] max abs diff = {max_diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export YOLOv8 -> ONNX with parity check.")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--no-dynamic", action="store_true", help="disable dynamic axes")
    parser.add_argument("--no-simplify", action="store_true", help="skip onnxsim")
    parser.add_argument("--half", action="store_true", help="export FP16 (GPU runtime)")
    parser.add_argument("--no-verify", action="store_true", help="skip parity check")
    args = parser.parse_args(argv)

    onnx_path = export(
        args.weights,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=not args.no_dynamic,
        simplify=not args.no_simplify,
        half=args.half,
    )

    if not args.no_verify:
        ok = verify_parity(args.weights, onnx_path, imgsz=args.imgsz)
        if not ok:
            print("[export] WARNING: parity check failed; inspect export settings.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
