"""evaluation_smoke for RAIM MEF Track 2."""

from __future__ import annotations

from typing import Any

from ltx_trainer.raim_mef.benchmarks import PAPER_ARXIV, TABLE1_FINAL_RESULTS, benchmarks_bundle
from ltx_trainer.raim_mef.metrics import leaderboard_score
from ltx_trainer.raim_mef.teams import team_cards


def evaluation_smoke() -> dict[str, Any]:
    whu = TABLE1_FINAL_RESULTS["WHU-VIP"]
    recomputed = leaderboard_score(whu["psnr_s1"], whu["ssim_s1"], whu["lpips_s1"])
    out: dict[str, Any] = {
        "package": "raim_mef",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "top_team": "WHU-VIP",
        "ref_final_score": whu["final_score"],
        "score_formula_check": round(recomputed, 3),
        "teams": list(team_cards().keys())[:4],
    }
    try:
        import torch

        from ltx_trainer.raim_mef.fusion import middle_exposure_fusion, weighted_fusion
        from ltx_trainer.raim_mef.model import RaimMefFusion
        from ltx_trainer.raim_mef.synthetic import TEST_EV_STOPS, synthesize_sequence

        scene = torch.rand(3, 64, 64)
        stack, gt, evs = synthesize_sequence(scene, ev_stops=TEST_EV_STOPS, shake_px=1.5)
        model = RaimMefFusion()
        model.eval()
        with torch.no_grad():
            pred = model(stack, evs)
        baseline = weighted_fusion(stack, evs)
        mid = middle_exposure_fusion(stack)

        model.train()
        loss, stats = model.training_step(stack, gt, evs)

        mse_base = float((baseline - gt).pow(2).mean())
        mse_pred = float((pred - gt).pow(2).mean())
        out.update(
            {
                "torch": True,
                "params_k": round(sum(p.numel() for p in model.parameters()) / 1000.0, 2),
                "loss_total": round(float(loss.item()), 5),
                "train_psnr": stats["psnr"],
                "train_leaderboard_score": round(stats["leaderboard_score"], 3),
                "pred_mse": round(mse_pred, 6),
                "baseline_mse": round(mse_base, 6),
                "mid_mse": round(float((mid - gt).pow(2).mean()), 6),
                "pred_beats_baseline": mse_pred < mse_base,
                "score_anchor_delta": round(abs(recomputed - whu["score_s1"]), 3),
            }
        )
    except ImportError:
        out["torch"] = False
    return out
