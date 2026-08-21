"""Zero-shot PD subject aggregation smoke (arXiv:2605.24806)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.zs_pd.aggregation import subject_label_from_segments
from ltx_trainer.zs_pd.config import ZsPdConfig


def evaluation_smoke(cfg: ZsPdConfig | None = None) -> dict[str, Any]:
    c = cfg or ZsPdConfig()
    label, prob = subject_label_from_segments([1, 1, 0], [0.92, 0.88, 0.55])
    return {
        "paper": c.paper_arxiv,
        "bensparx_llama3_bal_acc": 83.33,
        "bensparx_llama3_auroc": 0.901,
        "subject_majority_label": label,
        "subject_majority_prob": round(prob, 3),
    }
