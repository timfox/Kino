"""Framework card, demos, smoke for ShotCrop3 (arXiv:2606.05635)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.shotcrop3.config import ShotCrop3Config
from ltx_trainer.shotcrop3.paper_tables import (
    dataset_catalog,
    table1_gpt5,
    table1_ours,
    table1_rows,
    table2_ablation,
)
from ltx_trainer.shotcrop3.simulation import ablation_from_synthetic, benchmark_synthetic, pgs_demo, synthetic_scene


def framework_card(cfg: ShotCrop3Config | None = None) -> dict[str, Any]:
    cfg = cfg or ShotCrop3Config()
    p = cfg.params
    return {
        "name": "ShotCrop3",
        "paper": cfg.paper_arxiv,
        "title": "Cropping Human-Centric Images into Cinematic Triple-Shot Compositions",
        "task": "Triple-Shot Compositions (TSC): establishing + medium + close-up + captions",
        "pipeline": [
            "Stage 1 CoT-SFT on 7.6k expert triple-box annotations",
            "Stage 2 Semi-SFT with PGS pseudo labels (MLLM + CLIP + aesthetic)",
            "Stage 3 GRPO-S with IoU + aesthetic + aspect-ratio rewards",
        ],
        "shots": list(cfg.shot_types),
        "domains": list(cfg.dataset_domains),
        "train_test": (cfg.train_samples, cfg.test_samples),
        "base_model": cfg.base_model,
        "grpo_weights": {
            "lambda_iou": p.grpo.lambda_iou,
            "lambda_aes": p.grpo.lambda_aes,
            "lambda_ratio": p.grpo.lambda_ratio,
        },
        "pgs_thresholds": {"tau_high": p.pgs.tau_high, "tau_low": p.pgs.tau_low},
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub — no Qwen3-VL LoRA training or real TSC-Bench image renders.",
        "PGS uses synthetic score proxies, not Qwen3-VL-235B + rsinema/aesthetic-scorer.",
        "Table 1–2 values are paper anchors; synthetic ordering validates stage progression.",
        "CoT output format uses redacted_thinking + bbox tags as documented in appendix.",
        "Static human-centric images only; video extension noted as future work.",
    ]


def evaluation_demo(cfg: ShotCrop3Config | None = None) -> dict[str, Any]:
    cfg = cfg or ShotCrop3Config()
    scene = synthetic_scene(seed=11)
    syn = benchmark_synthetic(seed=7)
    pgs = pgs_demo(scene, seed=5, cfg=cfg)
    ablation = ablation_from_synthetic(seed=7)
    ours = table1_ours()
    gpt5 = table1_gpt5()
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "dataset": dataset_catalog(),
        "tables": {
            "table1": table1_rows(),
            "table2_ablation": table2_ablation(),
        },
        "synthetic_benchmark": syn,
        "pgs_demo": pgs,
        "ablation_synthetic": ablation,
        "iou_improvement_vs_gpt5": ours["iou"] / gpt5["iou"],
    }


def evaluation_smoke(cfg: ShotCrop3Config | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    ours = demo["tables"]["table1"]["ShotCrop3"]
    gpt5 = demo["tables"]["table1"]["GPT-5"]
    syn = demo["synthetic_benchmark"]
    ablation = demo["ablation_synthetic"]
    t2 = demo["tables"]["table2_ablation"]

    assert abs(ours["iou"] - 0.544) < 0.001
    assert abs(ours["bde"] - 0.087) < 0.001
    assert abs(ours["overall"] - 0.623) < 0.001
    assert demo["iou_improvement_vs_gpt5"] >= 2.82
    assert syn["grpo"]["iou"] > syn["base"]["iou"]
    assert syn["grpo"]["bde"] < syn["base"]["bde"]
    assert syn["semi_sft"]["iou"] > syn["cot_sft"]["iou"]
    assert syn["cot_sft"]["overall"] > syn["base"]["overall"]
    assert ablation[-1]["storytelling"] >= ablation[0]["overall"]
    assert t2[-1]["iou_val"] >= t2[2]["iou_val"]  # full GRPO >= +CoT-SFT

    return {
        "status": "ok",
        "paper": (cfg or ShotCrop3Config()).paper_arxiv,
        "table1_iou": ours["iou"],
        "gpt5_iou_ratio": demo["iou_improvement_vs_gpt5"],
        "synthetic_grpo_iou": syn["grpo"]["iou"],
        "demo_keys": list(demo.keys()),
    }
