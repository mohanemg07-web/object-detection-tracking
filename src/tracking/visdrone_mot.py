"""Parse VisDrone-MOT ground-truth annotation files.

Format (one line per object per frame):

    <frame>,<target_id>,<bbox_left>,<bbox_top>,<bbox_w>,<bbox_h>,
    <score>,<category>,<truncation>,<occlusion>

For MOTA evaluation we keep objects with ``score != 0`` (0 marks ignored
regions in MOT gt) and categories in the 10 detection classes, returning
boxes as ``(x1, y1, x2, y2)`` grouped by frame.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.data.visdrone import SOURCE_TO_YOLO


def parse_mot_line(line: str) -> tuple | None:
    """Parse one VisDrone-MOT gt line.

    Returns ``(frame, target_id, x1, y1, x2, y2, yolo_cls)`` or ``None`` if
    the line is blank, an ignored region (score 0), or a dropped category.
    """
    line = line.strip().rstrip(",")
    if not line:
        return None
    parts = line.split(",")
    if len(parts) < 8:
        raise ValueError(f"expected >=8 fields, got {len(parts)}: {line!r}")

    frame = int(float(parts[0]))
    target_id = int(float(parts[1]))
    left = float(parts[2])
    top = float(parts[3])
    width = float(parts[4])
    height = float(parts[5])
    score = float(parts[6])
    category = int(float(parts[7]))

    if score == 0:  # ignored region in MOT gt
        return None
    if category not in SOURCE_TO_YOLO:  # drop ignored(0)/others(11)/etc.
        return None
    if width <= 0 or height <= 0:
        return None

    return (frame, target_id, left, top, left + width, top + height, SOURCE_TO_YOLO[category])


def load_mot_gt(gt_path: Path) -> dict[int, list[tuple]]:
    """Load a VisDrone-MOT gt file into ``{frame: [(target_id, x1,y1,x2,y2), ...]}``."""
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"MOT gt not found: {gt_path}")

    by_frame: dict[int, list[tuple]] = defaultdict(list)
    for raw in gt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_mot_line(raw)
        if parsed is None:
            continue
        frame, tid, x1, y1, x2, y2, _cls = parsed
        by_frame[frame].append((tid, x1, y1, x2, y2))
    return dict(by_frame)
