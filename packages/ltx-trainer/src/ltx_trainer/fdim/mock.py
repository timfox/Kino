"""Runnable evaluation smoke for FDIM (arXiv:2604.24123)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fdim.benchmarks import PAPER_ARXIV, TABLE2_DCVQA_CODEC_GROUP, benchmarks_bundle


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "fdim",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_dcvqa_fdim_trad_plcc": TABLE2_DCVQA_CODEC_GROUP["fdim_traditional"]["plcc"],
    }
    try:
        import torch

        from ltx_trainer.fdim.mapping import LogisticMapping, fuse_component_scores
        from ltx_trainer.fdim.model import FDIM, FDIMConfig
        from ltx_trainer.fdim.ranking_loss import RankingLoss
        from ltx_trainer.fdim.synthetic import synthesize_distortion

        model = FDIM(FDIMConfig())
        model.eval()
        ref = torch.rand(3, 64, 64)
        dist_mild = synthesize_distortion(ref, 0.25)
        dist_heavy = synthesize_distortion(ref, 0.7)
        with torch.no_grad():
            q_clean = float(model(ref, ref))
            q_mild = float(model(ref, dist_mild))
            q_heavy = float(model(ref, dist_heavy))
            q_deep = float(model.deep_only(ref, dist_heavy))

        out.update(
            {
                "q_clean": round(q_clean, 4),
                "q_mild": round(q_mild, 4),
                "q_heavy": round(q_heavy, 4),
                "q_deep_only": round(q_deep, 4),
                "ordering_ok": q_clean > q_mild > q_heavy,
            }
        )

        loss_fn = RankingLoss()
        loss = loss_fn(
            torch.tensor([3.0]),
            torch.tensor([2.0]),
            torch.tensor([0.3]),
            torch.tensor([0.3]),
            torch.tensor([4.0]),
            torch.tensor([2.5]),
            torch.tensor([0.4]),
            torch.tensor([0.4]),
        )
        out["ranking_loss"] = round(float(loss.item()), 4)

        lm = LogisticMapping()
        fused = fuse_component_scores(lm(torch.tensor([2.0])), lm(torch.tensor([80.0])))
        out["fusion_scalar"] = round(float(fused.item()), 4)
        return out
    except ImportError:
        x = np.linspace(0, 1, 32, dtype=np.float64)
        out.update({"torch": False, "proxy_mse": round(float(np.mean((x - x * 0.7) ** 2)), 5)})
        return out
