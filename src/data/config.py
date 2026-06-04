"""Shared helpers for loading the paths config used across the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem paths for the data pipeline.

    All directories are absolute, derived from the repo root so the
    pipeline behaves the same regardless of the current working dir.
    """

    data_root: Path
    raw_dir: Path
    yolo_dir: Path
    det_dataset: str
    mot_dataset: str
    weights_dir: Path
    runs_dir: Path
    outputs_dir: Path
    data_yaml: Path

    @property
    def det_yolo_dir(self) -> Path:
        """Root of the converted YOLO-format detection dataset."""
        return self.yolo_dir / self.det_dataset


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def load_paths(config_path: str | Path = "configs/paths.yaml") -> Paths:
    """Load and resolve paths.yaml into an absolute, typed Paths object."""
    config_path = _resolve(REPO_ROOT, str(config_path))
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    return Paths(
        data_root=_resolve(REPO_ROOT, cfg["data_root"]),
        raw_dir=_resolve(REPO_ROOT, cfg["raw_dir"]),
        yolo_dir=_resolve(REPO_ROOT, cfg["yolo_dir"]),
        det_dataset=cfg["det_dataset"],
        mot_dataset=cfg["mot_dataset"],
        weights_dir=_resolve(REPO_ROOT, cfg["weights_dir"]),
        runs_dir=_resolve(REPO_ROOT, cfg["runs_dir"]),
        outputs_dir=_resolve(REPO_ROOT, cfg["outputs_dir"]),
        data_yaml=_resolve(REPO_ROOT, cfg["data_yaml"]),
    )
