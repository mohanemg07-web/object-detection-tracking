"""Tests for VisDrone-MOT ground-truth parsing."""

import pytest

from src.tracking.visdrone_mot import load_mot_gt, parse_mot_line


def test_parse_basic_mot_line():
    # frame, id, left, top, w, h, score, category, trunc, occ
    parsed = parse_mot_line("1,5,100,200,40,80,1,4,0,0")
    assert parsed is not None
    frame, tid, x1, y1, x2, y2, cls = parsed
    assert frame == 1
    assert tid == 5
    assert (x1, y1, x2, y2) == (100, 200, 140, 280)
    assert cls == 3  # car source 4 -> yolo 3


def test_parse_ignored_region_score_zero_dropped():
    assert parse_mot_line("1,5,100,200,40,80,0,4,0,0") is None


def test_parse_dropped_category():
    # category 11 (others) -> dropped
    assert parse_mot_line("1,5,100,200,40,80,1,11,0,0") is None
    # category 0 (ignored) -> dropped
    assert parse_mot_line("1,5,100,200,40,80,1,0,0,0") is None


def test_parse_blank_line():
    assert parse_mot_line("") is None
    assert parse_mot_line("   ") is None


def test_parse_too_few_fields_raises():
    with pytest.raises(ValueError):
        parse_mot_line("1,2,3")


def test_parse_degenerate_box_dropped():
    assert parse_mot_line("1,5,100,200,0,80,1,4,0,0") is None


def test_load_mot_gt_groups_by_frame(tmp_path):
    gt = tmp_path / "seq.txt"
    gt.write_text(
        "\n".join(
            [
                "1,1,10,10,20,20,1,4,0,0",  # frame 1, car
                "1,2,50,50,20,20,1,1,0,0",  # frame 1, pedestrian
                "2,1,12,12,20,20,1,4,0,0",  # frame 2, car
                "2,9,0,0,5,5,0,4,0,0",  # frame 2, ignored (score 0) -> dropped
            ]
        ),
        encoding="utf-8",
    )
    by_frame = load_mot_gt(gt)
    assert set(by_frame.keys()) == {1, 2}
    assert len(by_frame[1]) == 2
    assert len(by_frame[2]) == 1  # the score-0 row dropped
    # entries are (target_id, x1, y1, x2, y2)
    ids_f1 = {obj[0] for obj in by_frame[1]}
    assert ids_f1 == {1, 2}


def test_load_mot_gt_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mot_gt(tmp_path / "nope.txt")
