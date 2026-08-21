"""Run MUSTBENCH metrics on builtin QA predictions."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mustbench.metrics import hit_at_t, mc_accuracy, meteor_proxy, temporal_iou_f1
from ltx_trainer.mustbench.qa_samples import builtin_qa_samples


def run_eval_suite(samples=None) -> dict[str, Any]:
    samples = samples or builtin_qa_samples()
    tsg_preds, tsg_gold = [], []
    ltr_preds, ltr_gold = [], []
    gto_preds, gto_gold = [], []
    mtr_pred, mtr_gold = [], []
    tad_pairs: list[tuple[str, str]] = []

    for s in samples:
        if s.pred is None:
            continue
        if s.task == "TSG":
            tsg_preds.append(float(s.pred))
            tsg_gold.append(float(s.gold))
        elif s.task == "LTR":
            ltr_preds.append(str(s.pred))
            ltr_gold.append(str(s.gold))
        elif s.task == "GTO":
            gto_preds.append(str(s.pred))
            gto_gold.append(str(s.gold))
        elif s.task == "MTR":
            mtr_pred.append(tuple(s.pred[0]))
            mtr_gold.append(tuple(s.gold[0]))
        elif s.task == "TAD":
            tad_pairs.append((str(s.pred), str(s.gold)))

    mtr = temporal_iou_f1(mtr_pred, mtr_gold) if mtr_pred else {"iou": 0.0, "f1": 0.0}
    met = float(sum(meteor_proxy(p, g) for p, g in tad_pairs) / max(len(tad_pairs), 1))
    return {
        "tsg_hit3": hit_at_t(tsg_preds, tsg_gold, tolerance_s=3.0),
        "ltr_acc": mc_accuracy(ltr_preds, ltr_gold),
        "gto_acc": mc_accuracy(gto_preds, gto_gold),
        "mtr_iou": mtr["iou"],
        "mtr_f1": mtr["f1"],
        "tad_meteor_proxy": met,
        "n_samples": len(samples),
    }


def eval_suite_smoke() -> dict[str, Any]:
    out = run_eval_suite()
    return {"computed_eval": True, **out}
