"""Evaluation smoke for PanoEnv (arXiv:2602.21992)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoenv.benchmarks import TABLE3_GRPO, benchmarks_bundle
from ltx_trainer.panoenv.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    ours = next(r for r in TABLE3_GRPO if "GRPO-Balanced" in r["model"])
    out: dict[str, Any] = {
        "package": "panoenv",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_total_acc": ours["total"],
        "ref_oe_acc": ours["oe"],
    }
    try:
        import torch

        from ltx_trainer.panoenv.config import PanoEnvConfig
        from ltx_trainer.panoenv.erp_geometry import erp_pixel_to_spherical, spherical_to_cartesian
        from ltx_trainer.panoenv.panoenv_net import PanoEnvRLStub
        from ltx_trainer.panoenv.panoenv_qa import dataset_card
        from ltx_trainer.panoenv.pipeline import evaluation_demo_run
        from ltx_trainer.panoenv.rewards import total_reward
        from ltx_trainer.panoenv.stitch import cubemap_to_erp_stub

        cfg = PanoEnvConfig()
        demo = evaluation_demo_run(cfg, device="cpu")
        pano = cubemap_to_erp_stub(height=32, width=64)
        lam, phi = erp_pixel_to_spherical(
            torch.tensor([32.0]), torch.tensor([16.0]), width=64, height=32
        )
        pts = spherical_to_cartesian(torch.tensor([5.0]), lam, phi)
        r = total_reward(
            "<Reasoning>x</Reasoning><Answer>About 4.2 meters</Answer>",
            "About 4.0 meters",
            "distance",
        )
        model = PanoEnvRLStub(cfg, vocab_size=64)
        o = model(pano, torch.randint(0, 64, (1, 8)))
        card = dataset_card()
        out.update(
            {
                "torch": True,
                "demo_stage2_loss": demo["stage2"]["loss"],
                "distance_reward": r,
                "logits_shape": list(o["logits"].shape),
                "panoenv_qa_total": card["total_qa"],
                "cartesian_shape": list(pts.shape),
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
