"""A compact, dependency-light ByteTrack implementation.

ByteTrack's key idea: associate high-confidence detections first, then
recover objects from *low*-confidence detections in a second matching
pass against the still-unmatched tracks. This keeps occluded/blurred
objects alive and is what gives the MOTA boost over plain SORT.

This is pure numpy + (optional) ``lap`` for the linear assignment, so the
CPU demo doesn't need torch. It pairs with :class:`OnnxDetector`.

Config keys mirror configs/bytetrack.yaml: track_high_thresh,
track_low_thresh, new_track_thresh, track_buffer, match_thresh.
"""

from __future__ import annotations

import numpy as np

from src.tracking.kalman import KalmanBoxTracker


def xyxy_to_cxcyah(box: np.ndarray) -> np.ndarray:
    """(x1,y1,x2,y2) -> (cx, cy, aspect=w/h, h)."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def cxcyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    """(cx, cy, aspect, h) -> (x1,y1,x2,y2)."""
    cx, cy, a, h = state[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def linear_assignment(cost: np.ndarray, thresh: float):
    """Solve assignment on a cost matrix; return matches + unmatched indices.

    Uses ``lap`` if available, else a greedy fallback. ``cost`` entries
    above ``thresh`` are disallowed. Returns (matches, unmatched_a,
    unmatched_b) where matches is a list of (i, j) pairs.
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    try:
        import lap

        _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
        matches = [(i, int(x[i])) for i in range(len(x)) if x[i] >= 0]
        unmatched_a = [i for i in range(len(x)) if x[i] < 0]
        unmatched_b = [j for j in range(len(y)) if y[j] < 0]
        return matches, unmatched_a, unmatched_b
    except ImportError:
        return _greedy_assignment(cost, thresh)


def _greedy_assignment(cost: np.ndarray, thresh: float):
    """Greedy nearest-cost matching fallback when ``lap`` is unavailable."""
    matches: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = sorted(
        ((cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])),
        key=lambda t: t[0],
    )
    for c, i, j in pairs:
        if c > thresh or i in used_a or j in used_b:
            continue
        matches.append((i, j))
        used_a.add(i)
        used_b.add(j)
    unmatched_a = [i for i in range(cost.shape[0]) if i not in used_a]
    unmatched_b = [j for j in range(cost.shape[1]) if j not in used_b]
    return matches, unmatched_a, unmatched_b


class Track:
    """A single tracked object backed by a Kalman filter."""

    _kf = KalmanBoxTracker()
    _count = 0

    def __init__(self, box: np.ndarray, score: float, cls: int) -> None:
        Track._count += 1
        self.track_id = Track._count
        self.mean, self.cov = self._kf.initiate(xyxy_to_cxcyah(box))
        self.score = score
        self.cls = cls
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.state = "tentative"  # tentative -> confirmed -> lost

    @classmethod
    def reset_ids(cls) -> None:
        cls._count = 0

    def predict(self) -> None:
        self.mean, self.cov = self._kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, box: np.ndarray, score: float, cls: int) -> None:
        self.mean, self.cov = self._kf.update(self.mean, self.cov, xyxy_to_cxcyah(box))
        self.score = score
        self.cls = cls
        self.hits += 1
        self.time_since_update = 0
        self.state = "confirmed"

    @property
    def xyxy(self) -> np.ndarray:
        return cxcyah_to_xyxy(self.mean)


class ByteTracker:
    """Multi-object tracker implementing ByteTrack's two-stage association."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.high_thresh = cfg.get("track_high_thresh", 0.5)
        self.low_thresh = cfg.get("track_low_thresh", 0.1)
        self.new_track_thresh = cfg.get("new_track_thresh", 0.6)
        self.track_buffer = cfg.get("track_buffer", 30)
        self.match_thresh = cfg.get("match_thresh", 0.8)
        self.tracks: list[Track] = []
        Track.reset_ids()

    def _match(self, tracks: list[Track], boxes: np.ndarray):
        """IoU-match a track subset to boxes; return matches + leftovers."""
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        track_boxes = np.array([t.xyxy for t in tracks])
        cost = 1.0 - iou_matrix(track_boxes, boxes)
        return linear_assignment(cost, 1.0 - self.match_thresh)

    def update(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> list[Track]:
        """Advance the tracker one frame and return active confirmed tracks.

        ``boxes`` is (N,4) xyxy, ``scores`` (N,), ``classes`` (N,).
        """
        for t in self.tracks:
            t.predict()

        scores = np.asarray(scores, dtype=np.float64)
        high = scores >= self.high_thresh
        low = (scores >= self.low_thresh) & (~high)

        boxes_high, scores_high, cls_high = boxes[high], scores[high], classes[high]
        boxes_low, scores_low, cls_low = boxes[low], scores[low], classes[low]

        # --- Stage 1: match all tracks to high-confidence detections ---
        matches, u_tracks, u_dets = self._match(self.tracks, boxes_high)
        for ti, di in matches:
            self.tracks[ti].update(boxes_high[di], scores_high[di], int(cls_high[di]))

        # --- Stage 2: match remaining tracks to low-confidence detections ---
        remaining_tracks = [self.tracks[i] for i in u_tracks]
        matches_low, u_tracks2, _ = self._match(remaining_tracks, boxes_low)
        for ti, di in matches_low:
            remaining_tracks[ti].update(boxes_low[di], scores_low[di], int(cls_low[di]))

        # tracks still unmatched after both stages age out
        still_unmatched = {id(remaining_tracks[i]) for i in u_tracks2}

        # --- Spawn new tracks from unmatched high-confidence detections ---
        for di in u_dets:
            if scores_high[di] >= self.new_track_thresh:
                self.tracks.append(Track(boxes_high[di], float(scores_high[di]), int(cls_high[di])))

        # --- Drop stale tracks ---
        kept: list[Track] = []
        for t in self.tracks:
            if id(t) in still_unmatched and t.state != "confirmed":
                continue  # tentative + unmatched -> delete immediately
            if t.time_since_update > self.track_buffer:
                continue
            kept.append(t)
        self.tracks = kept

        return [t for t in self.tracks if t.state == "confirmed" and t.time_since_update == 0]
