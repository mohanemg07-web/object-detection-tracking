"""VisDrone dataset constants and annotation-line parsing.

VisDrone-DET annotation format (one object per line, CSV):

    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,
    <category>,<truncation>,<occlusion>

Category ids in the *source* files:

    0  ignored regions   (DROP)
    1  pedestrian
    2  people
    3  bicycle
    4  car
    5  van
    6  truck
    7  tricycle
    8  awning-tricycle
    9  bus
    10 motor
    11 others            (DROP)

We keep source ids 1..10 and remap them to contiguous YOLO ids 0..9.
"""

from __future__ import annotations

from dataclasses import dataclass

# YOLO class names in id order (0..9). Single source of truth, kept in
# sync with configs/data.yaml.
CLASS_NAMES: list[str] = [
    "pedestrian",  # 0  (source 1)
    "people",  # 1  (source 2)
    "bicycle",  # 2  (source 3)
    "car",  # 3  (source 4)
    "van",  # 4  (source 5)
    "truck",  # 5  (source 6)
    "tricycle",  # 6  (source 7)
    "awning-tricycle",  # 7  (source 8)
    "bus",  # 8  (source 9)
    "motor",  # 9  (source 10)
]

NUM_CLASSES = len(CLASS_NAMES)

# Source category id -> YOLO class id. Source 0 (ignored) and 11 (others)
# are intentionally absent so they get dropped during conversion.
SOURCE_TO_YOLO: dict[int, int] = {src: src - 1 for src in range(1, 11)}

# Categories we explicitly discard.
DROP_CATEGORIES: frozenset[int] = frozenset({0, 11})


@dataclass(frozen=True)
class VisDroneObject:
    """A single parsed VisDrone annotation line (source coordinate space)."""

    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    score: float
    category: int
    truncation: int
    occlusion: int

    @property
    def is_kept(self) -> bool:
        """True if this object maps to one of the 10 detection classes."""
        return self.category in SOURCE_TO_YOLO

    @property
    def yolo_class(self) -> int | None:
        """YOLO class id (0..9) or None if the object should be dropped."""
        return SOURCE_TO_YOLO.get(self.category)


def parse_annotation_line(line: str) -> VisDroneObject | None:
    """Parse one VisDrone-DET annotation line into a VisDroneObject.

    Returns ``None`` for blank lines. Raises ``ValueError`` on malformed
    lines (too few fields or non-numeric values) so callers can surface
    data-integrity problems instead of silently skipping them.
    """
    line = line.strip().rstrip(",")
    if not line:
        return None

    parts = line.split(",")
    if len(parts) < 6:
        raise ValueError(f"expected >=6 comma-separated fields, got {len(parts)}: {line!r}")

    try:
        left = float(parts[0])
        top = float(parts[1])
        width = float(parts[2])
        height = float(parts[3])
        score = float(parts[4])
        category = int(float(parts[5]))
        # truncation/occlusion are optional in some dumps; default to 0.
        truncation = int(float(parts[6])) if len(parts) > 6 and parts[6] != "" else 0
        occlusion = int(float(parts[7])) if len(parts) > 7 and parts[7] != "" else 0
    except ValueError as exc:  # non-numeric field
        raise ValueError(f"non-numeric field in line {line!r}: {exc}") from exc

    return VisDroneObject(
        bbox_left=left,
        bbox_top=top,
        bbox_width=width,
        bbox_height=height,
        score=score,
        category=category,
        truncation=truncation,
        occlusion=occlusion,
    )


def to_yolo_bbox(
    obj: VisDroneObject, img_width: int, img_height: int
) -> tuple[int, float, float, float, float] | None:
    """Convert a kept VisDrone object to a normalized YOLO label tuple.

    Returns ``(class_id, x_center, y_center, w, h)`` with all box values
    normalized to ``[0, 1]`` and clamped to the image bounds, or ``None``
    if the object is dropped or degenerate (zero area after clamping).
    """
    cls = obj.yolo_class
    if cls is None:
        return None
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"invalid image size: {img_width}x{img_height}")

    # Clamp the box to the image extent (some VisDrone boxes spill over).
    x1 = max(0.0, min(obj.bbox_left, img_width))
    y1 = max(0.0, min(obj.bbox_top, img_height))
    x2 = max(0.0, min(obj.bbox_left + obj.bbox_width, img_width))
    y2 = max(0.0, min(obj.bbox_top + obj.bbox_height, img_height))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None  # degenerate after clamping; drop it

    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    w_norm = bw / img_width
    h_norm = bh / img_height

    return cls, x_center, y_center, w_norm, h_norm
