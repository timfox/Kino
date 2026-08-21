"""Detection metrics and DAWN benchmark stats (Sec. V)."""

from __future__ import annotations

from ltx_trainer.cadenet.schema import Detection


def _match_tp(pred: list[Detection], gt: list[Detection], *, iou: float = 0.5) -> tuple[int, int, int]:
    tp = fp = 0
    matched = set()
    for p in pred:
        hit = False
        for j, g in enumerate(gt):
            if j in matched:
                continue
            if p.iou(g) >= iou:
                tp += 1
                matched.add(j)
                hit = True
                break
        if not hit:
            fp += 1
    fn = len(gt) - len(matched)
    return tp, fp, fn


def precision_recall_f1(pred: list[Detection], gt: list[Detection], *, iou: float = 0.5) -> tuple[float, float, float]:
    tp, fp, fn = _match_tp(pred, gt, iou=iou)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def recall_score(pred: list[Detection], gt: list[Detection], *, iou: float = 0.5) -> float:
    _, rec, _ = precision_recall_f1(pred, gt, iou=iou)
    return rec


def detection_f1(pred: list[Detection], gt: list[Detection], *, iou: float = 0.5) -> float:
    return precision_recall_f1(pred, gt, iou=iou)[2]


def delta_f1(c1: float, c2: float) -> float:
    return c2 - c1


def dawn_table_ii() -> dict[str, dict[str, float | int]]:
    """Table II paper-reported macro F1 deltas."""
    return {
        "snow": {"c1_f1": 0.750, "c2_f1": 0.773, "delta_f1": 0.023, "n": 204},
        "rain": {"c1_f1": 0.732, "c2_f1": 0.736, "delta_f1": 0.004, "n": 200},
        "sand": {"c1_f1": 0.716, "c2_f1": 0.716, "delta_f1": 0.000, "n": 323},
        "fog": {"c1_f1": 0.699, "c2_f1": 0.692, "delta_f1": -0.008, "n": 600},
        "macro": {"c1_f1": 0.716, "c2_f1": 0.717, "delta_f1": 0.001, "n": 1327},
    }
