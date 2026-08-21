"""Framework card and benchmark tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fsc_net.config import FscNetConfig
from ltx_trainer.fsc_net.ffc import ffc_demo
from ltx_trainer.fsc_net.losses import losses_demo
from ltx_trainer.fsc_net.progressive import progressive_demo


def framework_card(cfg: FscNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FscNetConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "framework": cfg.framework,
        "backbone": cfg.backbone,
        "datasets": {"train": cfg.dataset_train, "eval_zero_shot": cfg.dataset_eval},
        "demo_url": cfg.demo_url,
        "components": [
            "channel_wise_subband_tf_gridnet",
            "fast_fourier_convolution_global_context",
            "frequency_progressive_curriculum",
            "multi_stage_lsgan",
        ],
        "architecture": {
            "blocks": cfg.num_blocks,
            "subbands": cfg.num_subbands,
            "params_m": cfg.params_m,
            "macs_g": cfg.macs_g,
            "progressive_windows": list(cfg.progressive_windows),
        },
        "headline": {
            "lsd_4k": cfg.lsd_4k,
            "pesq_4k": cfg.pesq_4k,
            "pesq_ears": cfg.pesq_ears,
        },
    }


def table1_vctk() -> list[dict[str, Any]]:
    """Table I — VCTK BWE comparison."""
    rows_4k = [
        ("AP-BWE", 0.9553, 4.2556, 2.3199, 29.76, 17.87),
        ("BAE-Net lite", 0.9894, 4.1423, 2.5435, 0.57, 0.057),
        ("BAE-Net*", 0.9041, 4.2207, 2.5519, 17.41, 26.32),
        ("AERO", 0.9919, 4.2795, 2.2901, 21.66, 51.74),
        ("SFNet", 0.9200, None, None, 1.33, 0.88),
        ("FSC-Net", 0.8771, 4.3134, 2.8092, 1.54, 27.74),
    ]
    rows_16k = [
        ("AP-BWE", 0.7290, 4.3913, 4.5014, 29.76, 17.87),
        ("BAE-Net lite", 0.7220, 4.3117, 4.2986, 0.57, 0.057),
        ("BAE-Net*", 0.7135, 4.5028, 4.3831, 17.41, 26.32),
        ("AERO", 0.7889, 4.2667, 4.3035, 21.66, 51.74),
        ("SFNet", 0.7300, None, None, 1.33, 0.88),
        ("FSC-Net", 0.7048, 4.4681, 4.5279, 1.54, 27.74),
    ]

    def _pack(scenario: str, rows: list[tuple]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for model, lsd, nisqa, pesq, params, macs in rows:
            out.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "lsd": lsd,
                    "nisqa": nisqa,
                    "pesq": pesq,
                    "params_m": params,
                    "macs_g": macs,
                }
            )
        return out

    return _pack("4_khz_to_48_khz", rows_4k) + _pack("16_khz_to_48_khz", rows_16k)


def table2_ears() -> list[dict[str, Any]]:
    """Table II — EARS zero-shot generalization (16→48 kHz)."""
    return [
        {"model": "AP-BWE", "lsd": 1.4245, "nisqa": 3.6141, "pesq": 3.9589},
        {"model": "BAE-Net lite", "lsd": 1.3257, "nisqa": 3.8174, "pesq": 4.0249},
        {"model": "BAE-Net*", "lsd": 1.2235, "nisqa": 3.8023, "pesq": 4.1345},
        {"model": "AERO", "lsd": 1.2804, "nisqa": 3.8250, "pesq": 4.0387},
        {"model": "FSC-Net", "lsd": 1.2067, "nisqa": 3.9214, "pesq": 4.2988},
    ]


def table3_ablation() -> list[dict[str, Any]]:
    """Table III — component ablation (4 kHz → 48 kHz)."""
    return [
        {
            "model": "A: TF-GridNet-cws (Baseline)",
            "lsd": 0.8843,
            "nisqa": 4.2033,
            "pesq": 2.5219,
        },
        {"model": "B: + FFC", "lsd": 0.8857, "nisqa": 4.2412, "pesq": 2.7011},
        {
            "model": "C: + FFC + Progressive Learning",
            "lsd": 0.8771,
            "nisqa": 4.3134,
            "pesq": 2.8092,
        },
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_vctk": table1_vctk(),
        "table2_ears": table2_ears(),
        "table3_ablation": table3_ablation(),
    }


def headline_results(cfg: FscNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FscNetConfig()
    fsc_4k = next(r for r in table1_vctk() if r["model"] == "FSC-Net" and r["scenario"] == "4_khz_to_48_khz")
    bae_star = next(r for r in table1_vctk() if r["model"] == "BAE-Net*")
    ears = next(r for r in table2_ears() if r["model"] == "FSC-Net")
    return {
        "best_lsd_4k": fsc_4k["lsd"],
        "best_pesq_4k": fsc_4k["pesq"],
        "pesq_gain_vs_bae_star_4k": round(fsc_4k["pesq"] - bae_star["pesq"], 4),
        "best_pesq_ears": ears["pesq"],
        "params_m": cfg.params_m,
    }


def evaluation_demo(*, seed: int = 0, cfg: FscNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FscNetConfig()
    return {
        "framework": framework_card(cfg),
        "ffc": ffc_demo(seed=seed, cfg=cfg),
        "progressive": progressive_demo(seed=seed, cfg=cfg),
        "losses": losses_demo(seed=seed, cfg=cfg),
        "headline": headline_results(cfg),
    }
