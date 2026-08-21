"""ChildVox evaluation smoke (arXiv:2605.29257)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.childvox.config import ChildVoxConfig
from ltx_trainer.childvox.eval_suite import eval_suite_smoke
from ltx_trainer.childvox.pipeline import pipeline_demo


def evaluation_smoke(cfg: ChildVoxConfig | None = None) -> dict[str, Any]:
    c = cfg or ChildVoxConfig()
    demo = pipeline_demo(c, seed=42)
    suite = eval_suite_smoke(c, seed=42)
    return {
        "paper": c.paper_arxiv,
        "num_datasets": c.num_datasets,
        "balanced_total_samples": c.balanced_total_samples,
        "balanced_train_count": demo["balanced_total"],
        "myst_wer_whisper_large": c.myst_wer_whisper_large,
        "eval_macro_accuracy_pct": suite["macro_accuracy_pct"],
        "computed_eval": suite["computed_not_config"],
    }
