"""Entropy-Guided NMS — EG-NMS (Sec. III-F, Eq. 3–4)."""

from __future__ import annotations

from ltx_trainer.cadenet.pee import reliability_at_box
from ltx_trainer.cadenet.schema import Detection


def score_detections(
    dets_s: list[Detection],
    dets_q: list[Detection],
    r_map,
    *,
    img_h: int,
    img_w: int,
) -> list[Detection]:
    """Weight Thread S/Q confidences by patch reliability."""
    scaled: list[Detection] = []
    for d in dets_s:
        r = reliability_at_box(r_map, d.x1, d.y1, d.x2, d.y2)
        scaled.append(
            Detection(d.x1, d.y1, d.x2, d.y2, d.conf, d.cls, score=r * d.conf, source="S")
        )
    for d in dets_q:
        r = reliability_at_box(r_map, d.x1, d.y1, d.x2, d.y2)
        scaled.append(
            Detection(d.x1, d.y1, d.x2, d.y2, d.conf, d.cls, score=(1.0 - r) * d.conf, source="Q")
        )
    return nms(scaled, iou_thresh=0.45)


def nms(dets: list[Detection], *, iou_thresh: float = 0.45) -> list[Detection]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    keep: list[Detection] = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if best.iou(d) < iou_thresh]
    return keep
