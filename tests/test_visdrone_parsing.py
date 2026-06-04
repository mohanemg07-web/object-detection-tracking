"""Unit tests for VisDrone annotation parsing and YOLO conversion."""

import pytest

from src.data.visdrone import (
    CLASS_NAMES,
    DROP_CATEGORIES,
    NUM_CLASSES,
    SOURCE_TO_YOLO,
    parse_annotation_line,
    to_yolo_bbox,
)


def test_class_mapping_is_contiguous_and_complete():
    assert NUM_CLASSES == 10
    assert len(CLASS_NAMES) == 10
    # source ids 1..10 -> yolo 0..9
    assert SOURCE_TO_YOLO == {s: s - 1 for s in range(1, 11)}
    assert set(SOURCE_TO_YOLO.values()) == set(range(10))
    assert DROP_CATEGORIES == {0, 11}


def test_parse_blank_line_returns_none():
    assert parse_annotation_line("") is None
    assert parse_annotation_line("   ") is None


def test_parse_basic_line():
    # left, top, w, h, score, category, trunc, occ
    obj = parse_annotation_line("100,200,50,80,1,4,0,0")
    assert obj is not None
    assert obj.bbox_left == 100
    assert obj.bbox_top == 200
    assert obj.bbox_width == 50
    assert obj.bbox_height == 80
    assert obj.category == 4  # car (source)
    assert obj.is_kept is True
    assert obj.yolo_class == 3  # car (yolo)


def test_parse_trailing_comma_tolerated():
    obj = parse_annotation_line("0,0,10,10,1,2,0,0,")
    assert obj is not None
    assert obj.category == 2


def test_parse_missing_optional_fields_defaults_zero():
    obj = parse_annotation_line("0,0,10,10,1,2")
    assert obj is not None
    assert obj.truncation == 0
    assert obj.occlusion == 0


def test_parse_too_few_fields_raises():
    with pytest.raises(ValueError):
        parse_annotation_line("1,2,3")


def test_parse_non_numeric_raises():
    with pytest.raises(ValueError):
        parse_annotation_line("a,b,c,d,e,f,g,h")


@pytest.mark.parametrize("dropped_cat", [0, 11])
def test_dropped_categories_have_no_yolo_class(dropped_cat):
    obj = parse_annotation_line(f"0,0,10,10,1,{dropped_cat},0,0")
    assert obj is not None
    assert obj.is_kept is False
    assert obj.yolo_class is None


def test_to_yolo_bbox_center_normalization():
    # 100x100 image, box at (10,20) size 40x60 -> center (30,50)
    obj = parse_annotation_line("10,20,40,60,1,4,0,0")
    result = to_yolo_bbox(obj, 100, 100)
    assert result is not None
    cls, xc, yc, w, h = result
    assert cls == 3
    assert xc == pytest.approx(0.30)
    assert yc == pytest.approx(0.50)
    assert w == pytest.approx(0.40)
    assert h == pytest.approx(0.60)


def test_to_yolo_bbox_clamps_overflow():
    # box spills past the right/bottom edge; should clamp to image bounds
    obj = parse_annotation_line("90,90,40,40,1,4,0,0")
    result = to_yolo_bbox(obj, 100, 100)
    assert result is not None
    _, xc, yc, w, h = result
    # clamped box is from (90,90) to (100,100): w=h=10, center (95,95)
    assert xc == pytest.approx(0.95)
    assert yc == pytest.approx(0.95)
    assert w == pytest.approx(0.10)
    assert h == pytest.approx(0.10)
    assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0


def test_to_yolo_bbox_dropped_category_returns_none():
    obj = parse_annotation_line("0,0,10,10,1,0,0,0")  # ignored region
    assert to_yolo_bbox(obj, 100, 100) is None


def test_to_yolo_bbox_degenerate_box_returns_none():
    # box entirely outside the image -> zero area after clamping
    obj = parse_annotation_line("200,200,10,10,1,4,0,0")
    assert to_yolo_bbox(obj, 100, 100) is None


def test_to_yolo_bbox_invalid_image_size_raises():
    obj = parse_annotation_line("0,0,10,10,1,4,0,0")
    with pytest.raises(ValueError):
        to_yolo_bbox(obj, 0, 100)
