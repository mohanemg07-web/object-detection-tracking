"""CLI to preview the preprocessing pipeline on a single image.

Usage:
    python -m src.preprocessing.cli --image path/to/img.jpg \
        --config configs/preprocess.yaml --out outputs/preprocess_demo.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

from src.preprocessing.transforms import build_pipeline
from src.preprocessing.visualize import visualize_pipeline


def load_preprocess_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview preprocessing on an image.")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--out", default="outputs/preprocess_demo.jpg")
    args = parser.parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        print(f"could not read image: {args.image}")
        return 1

    config = load_preprocess_config(args.config)
    pipeline = build_pipeline(config)
    print(f"pipeline has {len(pipeline)} transform(s)")

    visualize_pipeline(image, pipeline, save_path=args.out)
    print(f"wrote before/after comparison to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
