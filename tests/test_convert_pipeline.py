"""End-to-end test of the VisDrone->YOLO conversion on synthetic data.

Builds a tiny fake VisDrone DET tree (a couple of solid-color images plus
matching annotation files), runs the split conversion, then validates the
produced YOLO labels. No real dataset or GPU needed.
"""

from collections import Counter

import numpy as np
import pytest

from src.data.convert_visdrone import convert_split
from src.data.validate_labels import validate_split

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _make_image(path, w=200, h=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_ann(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_fake_visdrone(root):
    img_dir = root / "images"
    ann_dir = root / "annotations"
    # image 0001: a car (cat 4) + an ignored region (cat 0, dropped)
    _make_image(img_dir / "0001.jpg")
    _write_ann(ann_dir / "0001.txt", ["10,20,40,30,1,4,0,0", "0,0,200,100,0,0,0,0"])
    # image 0002: a pedestrian (cat 1) + 'others' (cat 11, dropped)
    _make_image(img_dir / "0002.jpg")
    _write_ann(ann_dir / "0002.txt", ["50,10,20,40,1,1,0,0", "0,0,5,5,1,11,0,0"])
    # image 0003: no annotation file -> background image
    _make_image(img_dir / "0003.jpg")


def test_convert_and_validate_roundtrip(tmp_path):
    src = tmp_path / "VisDrone2019-DET-train"
    out = tmp_path / "yolo" / "VisDrone-DET"
    _build_fake_visdrone(src)

    counter: Counter = Counter()
    stats = convert_split(src, out, lambda _stem: "train", counter)

    assert stats["train_images"] == 3
    # two kept boxes total (car + pedestrian); two dropped (ignored + others)
    assert stats["train_boxes"] == 2
    assert stats["dropped_boxes"] == 2
    assert counter[3] == 1  # car -> yolo 3
    assert counter[0] == 1  # pedestrian -> yolo 0

    # images and labels were written
    assert (out / "images" / "train" / "0001.jpg").exists()
    assert (out / "labels" / "train" / "0001.txt").exists()
    # background image gets an (empty) label file
    bg = out / "labels" / "train" / "0003.txt"
    assert bg.exists()
    assert bg.read_text(encoding="utf-8").strip() == ""

    # validator should pass cleanly on the converted output
    res = validate_split(out / "images" / "train", out / "labels" / "train")
    assert res["present"] is True
    assert res["errors"] == []
    assert res["n_images"] == 3
    assert res["n_boxes"] == 2
    assert res["n_empty"] == 1
