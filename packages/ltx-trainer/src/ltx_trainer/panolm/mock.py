"""Runnable evaluation smoke for PanoLM / PanoVQA (arXiv:2603.09573)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panolm.benchmarks import TABLE5_PANOVQA, benchmarks_bundle
from ltx_trainer.panolm.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    ours = next(r for r in TABLE5_PANOVQA if "PanoLM" in r["method"])
    out: dict[str, Any] = {
        "package": "panolm",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_avg_gpt": ours["avg"],
    }
    try:
        import torch

        from ltx_trainer.panolm.config import PanoLMConfig
        from ltx_trainer.panolm.panolm_net import PanoLMStub
        from ltx_trainer.panolm.pha import PanoramicHybridBlock
        from ltx_trainer.panolm.pipeline import evaluation_demo_run
        from ltx_trainer.panolm.panovqa import dataset_card

        cfg = PanoLMConfig(height=56, width=112)
        demo = evaluation_demo_run(cfg, device="cpu")
        model = PanoLMStub(cfg)
        img = torch.rand(1, 3, 56, 112)
        o = model(img, torch.randint(0, 500, (1, 8)))
        pha = PanoramicHybridBlock(cfg.embed_dim, window_size=8, top_k=16)
        tokens = torch.randn(1, 32, cfg.embed_dim)
        t2 = pha(tokens)
        card = dataset_card()
        out.update(
            {
                "torch": True,
                "demo_loss": demo["train"]["loss"],
                "logits_shape": list(o["logits"].shape),
                "pha_shape": list(t2.shape),
                "panovqa_total": card["total_qa"],
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
