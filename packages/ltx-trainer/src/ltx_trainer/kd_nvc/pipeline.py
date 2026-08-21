"""Framework card, demos, smoke for KD-NVC (arXiv:2606.04595)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.kd_nvc.ae_nas import search_space, select_student
from ltx_trainer.kd_nvc.config import KDNVCConfig, KD_NVC_S, KD_NVC_T
from ltx_trainer.kd_nvc.metrics import (
    fig1_gain_anchors,
    operating_guidelines,
    table_bd_rate_ip32,
    table_bd_rate_ip_minus1,
    table_complexity,
    table_distill_cost,
    table_fps_rtx5060_1080p,
)
from ltx_trainer.kd_nvc.simulation import (
    ae_nas_vs_uniform,
    compare_distillation_at_speedup,
    efd_vs_mse_rd_loss,
    selected_students,
)


def framework_card(cfg: KDNVCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or KDNVCConfig()
    return {
        "name": "KD-NVC",
        "paper": cfg.paper_arxiv,
        "title": "Search-and-distill framework to accelerate neural video coding",
        "teacher": cfg.teacher,
        "anchor": cfg.anchor,
        "stages": ["AE-NAS (module Pareto + η̂ selection)", "One-step EFD distillation"],
        "students": [KD_NVC_S.name, KD_NVC_T.name],
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub — no DCVC-RT training loop or fvcore MAC counts at runtime.",
        "BD-rate and FPS values use paper anchors + phenomenological ordering checks.",
        "AE-NAS search space size (~80) is modeled, not exhaustively re-trained.",
        "EFD uses synthetic sparse features, not Vimeo-90k patches.",
        "Entropy model excluded from architecture search as in §IV-B.",
    ]


def evaluation_demo(cfg: KDNVCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or KDNVCConfig()
    dist_60 = compare_distillation_at_speedup(speedup_pct=60.0)
    dist_100 = compare_distillation_at_speedup(speedup_pct=100.0)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "guidelines": operating_guidelines(),
        "ae_nas": {
            "search_space_size": len(search_space()),
            "selected_s": select_student(60.0).__dict__,
            "selected_t": select_student(100.0).__dict__,
            "students": selected_students(),
            "vs_uniform_60": ae_nas_vs_uniform(60.0),
        },
        "distillation_60pct": {k: v.__dict__ for k, v in dist_60.items()},
        "distillation_100pct": {k: v.__dict__ for k, v in dist_100.items()},
        "efd_ablation": efd_vs_mse_rd_loss(),
        "tables": {
            "bd_rate_ip32": table_bd_rate_ip32(),
            "bd_rate_ip_minus1": table_bd_rate_ip_minus1(),
            "complexity": table_complexity(),
            "fps_5060_1080p": table_fps_rtx5060_1080p(),
            "distill_cost": table_distill_cost(),
            "fig1": fig1_gain_anchors(),
        },
    }


def evaluation_smoke(cfg: KDNVCConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    t3 = demo["tables"]["bd_rate_ip32"]
    d60 = demo["distillation_60pct"]
    d100 = demo["distillation_100pct"]
    uni = demo["ae_nas"]["vs_uniform_60"]
    efd = demo["efd_ablation"]
    fps = demo["tables"]["fps_5060_1080p"]

    assert d60["kd_nvc"]["avg_bd_rate_ip32"] < d60["smodi"]["avg_bd_rate_ip32"]
    assert d60["kd_nvc"]["avg_bd_rate_ip32"] == t3["speedup_60pct"]["kd_nvc_s_avg"]
    assert d100["kd_nvc"]["avg_bd_rate_ip32"] < d100["smodi"]["avg_bd_rate_ip32"]
    assert d100["kd_nvc"]["avg_bd_rate_ip32"] == t3["speedup_100pct"]["kd_nvc_t_avg"]
    assert uni["ae_nas_eta"] > uni["uniform_eta"]
    assert uni["ae_nas_bd_rate"] < uni["uniform_bd_rate"]
    assert efd["efd_total_loss"] < efd["mse_rd_loss"]
    assert fps["kd_nvc_t_decode_fps"] >= fps["high_framerate_fps"]
    assert demo["ae_nas"]["search_space_size"] >= 20

    return {
        "status": "ok",
        "paper": (cfg or KDNVCConfig()).paper_arxiv,
        "kd_nvc_s_bd_ip32": d60["kd_nvc"]["avg_bd_rate_ip32"],
        "decode_fps_5060": fps["kd_nvc_t_decode_fps"],
        "demo_keys": list(demo.keys()),
    }
