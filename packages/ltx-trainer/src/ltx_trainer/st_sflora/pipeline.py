"""Framework card and benchmark bundles for ST-SFLora."""

from __future__ import annotations

from typing import Any

from ltx_trainer.st_sflora.config import StSfloraConfig
from ltx_trainer.st_sflora.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.st_sflora.mock import evaluation_smoke
from ltx_trainer.st_sflora.tables import (
    fig8_ste_ablation_labels,
    headline_results,
    table1_top1_accuracy,
    table2_client_overhead,
)


def framework_card(cfg: StSfloraConfig | None = None) -> dict[str, Any]:
    cfg = cfg or StSfloraConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "problem": (
            "Split federated LoRA for ViTs on mobile edge: uplink activation tokens "
            "dominate cost. ST-SFLora selects CLS-relevant patches via attention, merges "
            "discards, and jointly optimizes K, bandwidth, and power under STE."
        ),
        "architecture": {
            "paradigm": "one_way_uplink_sfl_lora",
            "client": f"embedding + first {cfg.client_cut_layers} ViT blocks (frozen)",
            "server": "remaining blocks + LoRA adapters (trainable)",
            "no_client_gradients": True,
        },
        "ste": {
            "formula": "E = sum_m f_m(K_m) / max_m T^U_m",
            "f_m": "cumulative batch token importance (concave in K)",
        },
        "optimization": {
            "problem": "P0 maximize STE subject to power/bandwidth/token integer constraints",
            "solver": "Alternating SUBP1 power, SUBP2 bandwidth/tau, SUBP3 K_max (Alg. 4)",
        },
        "wireless_defaults": {
            "clients": cfg.num_clients,
            "bandwidth_mhz": cfg.bandwidth_total_hz / 1e6,
            "p_max_w": cfg.p_max_w,
            "batch_size": cfg.batch_size,
            "patch_tokens": cfg.num_patch_tokens,
        },
        "benchmarks": list(cfg.benchmarks),
        "backbones": list(cfg.backbones),
        "baselines": list(cfg.baselines),
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_top1_accuracy": table1_top1_accuracy(),
        "table2_client_overhead": table2_client_overhead(),
        "fig8_ste_ablations": fig8_ste_ablation_labels(),
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return evaluation_smoke()
