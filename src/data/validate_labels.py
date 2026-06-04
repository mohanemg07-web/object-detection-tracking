"""Validate the converted YOLO dataset and print a dataset summary.

Checks performed per split (train/val/test):
  * every image has a corresponding label file (empty allowed),
  * every label line has exactly 5 fields,
  * class ids are integers in 0..9,
  * box center/size are floats normalized to [0, 1],
  * boxes have positive area.

Also prints per-split image/box counts and a per-class histogram so you
can eyeball class balance. Exits non-zero if any integrity error is found.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.data.config import load_paths
from src.data.visdrone import CLASS_NAMES, NUM_CLASSES

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def validate_label_line(line: str) -> tuple[int | None, str | None]:
    """Validate a single YOLO label line.

    Returns (class_id, error). On success error is None; on failure
    class_id is None and error describes the problem.
    """
    parts = line.split()
    if len(parts) != 5:
        return None, f"expected 5 fields, got {len(parts)}: {line!r}"
    try:
        cls = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None, f"non-numeric field: {line!r}"

    if not (0 <= cls < NUM_CLASSES):
        return None, f"class id {cls} out of range 0..{NUM_CLASSES - 1}"

    xc, yc, w, h = coords
    for label, value in (("xc", xc), ("yc", yc), ("w", w), ("h", h)):
        if not (0.0 <= value <= 1.0):
            return None, f"{label}={value} not in [0,1]: {line!r}"
    if w <= 0 or h <= 0:
        return None, f"non-positive box size w={w} h={h}: {line!r}"

    return cls, None


def validate_split(images_dir: Path, labels_dir: Path) -> dict:
    """Validate one split, returning a result dict with counts and errors."""
    errors: list[str] = []
    class_counter: Counter = Counter()
    n_images = 0
    n_boxes = 0
    n_empty = 0

    if not images_dir.is_dir():
        return {"present": False, "errors": [f"missing images dir: {images_dir}"]}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        n_images += 1
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"missing label for image: {img_path.name}")
            continue
        lines = [ln for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            n_empty += 1
        for ln in lines:
            cls, err = validate_label_line(ln)
            if err is not None:
                errors.append(f"{label_path.name}: {err}")
            else:
                class_counter[cls] += 1
                n_boxes += 1

    return {
        "present": True,
        "n_images": n_images,
        "n_boxes": n_boxes,
        "n_empty": n_empty,
        "class_counter": class_counter,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate converted YOLO labels.")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument(
        "--max-errors", type=int, default=20, help="how many errors to print per split"
    )
    args = parser.parse_args(argv)

    paths = load_paths(args.config)
    root = paths.det_yolo_dir

    if not root.is_dir():
        print(f"Converted dataset not found at {root}. Run convert_visdrone first.")
        return 1

    total_errors = 0
    grand_class: Counter = Counter()
    print(f"=== Validating {root} ===\n")

    for split in ("train", "val", "test"):
        res = validate_split(root / "images" / split, root / "labels" / split)
        if not res.get("present"):
            print(f"[{split}] not present: {res['errors']}")
            continue
        errs = res["errors"]
        total_errors += len(errs)
        grand_class.update(res["class_counter"])
        status = "OK" if not errs else f"{len(errs)} ERRORS"
        print(
            f"[{split}] {res['n_images']} images, {res['n_boxes']} boxes, "
            f"{res['n_empty']} empty/background — {status}"
        )
        for e in errs[: args.max_errors]:
            print(f"    - {e}")
        if len(errs) > args.max_errors:
            print(f"    ... and {len(errs) - args.max_errors} more")

    print("\n  per-class totals (all splits):")
    for cid, name in enumerate(CLASS_NAMES):
        print(f"    {cid} {name:16s}: {grand_class.get(cid, 0):8d}")

    if total_errors:
        print(f"\nFAILED: {total_errors} integrity error(s) found.")
        return 1
    print("\nPASSED: dataset integrity checks clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
