"""Fine-tune YOLOv8m on VisDrone-DET with MLflow (DagsHub) logging.

GPU-only in practice: YOLOv8m fine-tuning on ~10k images is impractical
on CPU. We detect CUDA at runtime and refuse to start a real run on CPU
unless ``--allow-cpu`` is passed (useful for a 1-epoch smoke test).

What gets logged to MLflow:
  * all resolved hyperparameters (params),
  * per-epoch metrics (mAP@0.5, mAP@0.5:0.95, losses) via a callback,
  * final metrics, the results/confusion-matrix/PR plots, and
  * best.pt + last.pt weights as artifacts.

Run:
    python -m src.training.train --config configs/train.yaml
    python -m src.training.train --config configs/train.yaml --epochs 1 --allow-cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.training.mlflow_utils import resolve_tracking_uri, setup_mlflow

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def detect_device(requested: str | int, allow_cpu: bool) -> str:
    """Return a YOLO device string, honoring CUDA availability.

    Fails fast on CPU-only machines unless ``allow_cpu`` is set, so users
    don't accidentally kick off a multi-hour CPU run.
    """
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False

    if cuda:
        return str(requested)

    if not allow_cpu:
        raise SystemExit(
            "No CUDA GPU detected. YOLOv8m fine-tuning needs an NVIDIA GPU.\n"
            "  * Run this on a free Colab T4 (see notebooks/train_colab.ipynb), or\n"
            "  * pass --allow-cpu --epochs 1 for a tiny CPU smoke test."
        )
    print("[device] CUDA not available; running on CPU (smoke test only).")
    return "cpu"


def _build_overrides(cfg: dict, args: argparse.Namespace, device: str) -> dict:
    """Merge config + CLI overrides into Ultralytics train() kwargs."""
    train_keys = (
        "data",
        "epochs",
        "imgsz",
        "batch",
        "patience",
        "optimizer",
        "lr0",
        "lrf",
        "weight_decay",
        "warmup_epochs",
        "seed",
        "cos_lr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "fliplr",
        "mosaic",
        "close_mosaic",
        "workers",
        "project",
        "name",
        "exist_ok",
    )
    overrides = {k: cfg[k] for k in train_keys if k in cfg}
    # CLI overrides take precedence when provided.
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.data is not None:
        overrides["data"] = args.data
    overrides["device"] = device
    return overrides


def _make_mlflow_callback(mlflow):
    """Build an Ultralytics on_fit_epoch_end callback that logs to MLflow."""

    def _on_fit_epoch_end(trainer):
        try:
            metrics = {
                k.replace("(", "").replace(")", ""): float(v) for k, v in trainer.metrics.items()
            }
            mlflow.log_metrics(metrics, step=trainer.epoch)
        except Exception as exc:  # noqa: BLE001
            print(f"[mlflow] epoch log skipped: {exc}")

    return _on_fit_epoch_end


def train(cfg: dict, args: argparse.Namespace) -> Path | None:
    """Run training; return the path to best.pt (or None on CPU smoke runs)."""
    from ultralytics import YOLO

    device = detect_device(cfg.get("device", 0), args.allow_cpu)
    overrides = _build_overrides(cfg, args, device)

    tracking_uri = resolve_tracking_uri(args.mlflow_uri or cfg.get("mlflow_tracking_uri", ""))
    mlflow = setup_mlflow(tracking_uri, cfg.get("mlflow_experiment", "visdrone-yolov8m"))

    model = YOLO(cfg.get("model", "yolov8m.pt"))

    run_ctx = mlflow.start_run() if mlflow else None
    try:
        if mlflow:
            mlflow.log_params(overrides)
            mlflow.log_param("base_model", cfg.get("model", "yolov8m.pt"))
            model.add_callback("on_fit_epoch_end", _make_mlflow_callback(mlflow))

        results = model.train(**overrides)

        save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
        best = save_dir / "weights" / "best.pt" if save_dir else None

        if mlflow:
            _log_final_artifacts(mlflow, model, save_dir, best)

        if best and best.exists():
            print(f"\n[done] best weights: {best}")
        return best
    finally:
        if run_ctx is not None:
            mlflow.end_run()


def _log_final_artifacts(mlflow, model, save_dir: Path | None, best: Path | None) -> None:
    """Log final metrics, plots, and weights to MLflow (best-effort)."""
    try:
        final = getattr(model, "metrics", None)
        if final and getattr(final, "results_dict", None):
            metrics = {
                k.replace("(", "").replace(")", ""): float(v)
                for k, v in final.results_dict.items()
                if isinstance(v, (int, float))
            }
            mlflow.log_metrics(metrics)
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] final metrics skipped: {exc}")

    if save_dir and save_dir.exists():
        for plot in ("confusion_matrix.png", "results.png", "PR_curve.png", "val_batch0_pred.jpg"):
            p = save_dir / plot
            if p.exists():
                try:
                    mlflow.log_artifact(str(p), artifact_path="plots")
                except Exception as exc:  # noqa: BLE001
                    print(f"[mlflow] artifact {plot} skipped: {exc}")

    if best and best.exists():
        try:
            mlflow.log_artifact(str(best), artifact_path="weights")
            last = best.with_name("last.pt")
            if last.exists():
                mlflow.log_artifact(str(last), artifact_path="weights")
        except Exception as exc:  # noqa: BLE001
            print(f"[mlflow] weights artifact skipped: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8m on VisDrone-DET.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--data", default=None, help="override path to data.yaml")
    parser.add_argument("--mlflow-uri", default=None, help="override DagsHub MLflow URI")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run (for a tiny smoke test only)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    train(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
