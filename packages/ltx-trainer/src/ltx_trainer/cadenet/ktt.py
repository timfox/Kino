"""Kalman Temporal Tracker — KTT (Sec. III-G)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.cadenet.schema import Detection, Track


@dataclass
class KTTConfig:
    iou_threshold: float = 0.3
    max_misses: int = 3
    min_hits: int = 1
    conf_smooth: float = 0.7
    kalman_steps: int = 3


def _box_center(box: Detection) -> tuple[float, float, float, float]:
    w = max(box.x2 - box.x1, 1e-6)
    h = max(box.y2 - box.y1, 1e-6)
    cx = (box.x1 + box.x2) * 0.5
    cy = (box.y1 + box.y2) * 0.5
    return cx, cy, w / h, h


def predict_tracks(tracks: list[Track], *, steps: int = 1) -> list[Track]:
    """k-step constant-velocity Kalman projection."""
    out: list[Track] = []
    for t in tracks:
        st = t.kalman_state[:]
        for _ in range(steps):
            st[0] += st[4]
            st[1] += st[5]
            st[2] += st[6]
        cx, cy, a, h = st[0], st[1], st[2], st[3]
        w = a * h
        box = Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, t.box.conf, t.box.cls)
        out.append(Track(t.track_id, box, t.smoothed_conf, t.misses, t.hits, st))
    return out


def update_tracks(
    tracks: list[Track],
    detections: list[Detection],
    *,
    cfg: KTTConfig | None = None,
    next_id: int = 0,
) -> tuple[list[Track], int]:
    cfg = cfg or KTTConfig()
    if not detections:
        kept = []
        for t in tracks:
            t.misses += 1
            if t.misses <= cfg.max_misses:
                kept.append(t)
        return kept, next_id

    unmatched_dets = list(detections)
    updated: list[Track] = []
    for t in tracks:
        best_i, best_iou = -1, cfg.iou_threshold
        for i, d in enumerate(unmatched_dets):
            iou = t.box.iou(d)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0:
            d = unmatched_dets.pop(best_i)
            cx, cy, a, h = _box_center(d)
            st = t.kalman_state[:]
            st[4] = cx - st[0]
            st[5] = cy - st[1]
            st[0], st[1], st[2], st[3] = cx, cy, a, h
            sm = cfg.conf_smooth * d.conf + (1 - cfg.conf_smooth) * t.smoothed_conf
            updated.append(Track(t.track_id, d, sm, 0, t.hits + 1, st))
        else:
            t.misses += 1
            if t.misses <= cfg.max_misses:
                updated.append(t)
    for d in unmatched_dets:
        cx, cy, a, h = _box_center(d)
        st = [cx, cy, a, h, 0.0, 0.0, 0.0]
        updated.append(Track(next_id, d, d.conf, 0, 1, st))
        next_id += 1
    return [t for t in updated if t.hits >= cfg.min_hits], next_id
