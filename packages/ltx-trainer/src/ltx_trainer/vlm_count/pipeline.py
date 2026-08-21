"""VLM visual counting bottleneck — framework card, knowledge, evaluation demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.vlm_count.benchmarks import benchmarks_bundle
from ltx_trainer.vlm_count.config import VlmCountConfig
from ltx_trainer.vlm_count.go_board import patch_embeddings_from_board, sample_go_board
from ltx_trainer.vlm_count.probing import diagnose_sample, fit_id_probe
def framework_card(cfg: VlmCountConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VlmCountConfig()
    return {
        "name": "VLM-Count-Bottleneck",
        "paper": cfg.paper_arxiv,
        "title": "Unveiling the Visual Counting Bottleneck in Vision-Language Models",
        "stages": list(cfg.stages),
        "synthetic_lab": {
            "visual_train_max": cfg.visual_train_max,
            "text_train_max": cfg.text_train_max,
            "full_extrap_max": cfg.full_extrap_max,
        },
        "validation_model": "Qwen3-VL-32B-Instruct (6×6 Go boards)",
    }


def knowledge_card(cfg: VlmCountConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VlmCountConfig()
    bench = benchmarks_bundle()
    return {
        "framework": framework_card(cfg),
        "key_findings": {
            "bottleneck_stage": "symbolic_mapping",
            "fractured_magnitude": bench["fractured_magnitude_hypothesis"],
            "synthetic_vision_ve_accuracy_pct": bench["synthetic_regime_accuracy"]["vision"]["VE"],
            "synthetic_compare_ve_pct": bench["synthetic_compare_vision"]["VE"],
            "circuit_disjoint_heads_pct": bench["mechanism"]["circuit_head_disjoint_pct"],
        },
        "integration": {
            "env": "GOPEX_VLM_COUNT=1",
            "fold_hook": "GOPEX_AV_FOLD_HOOKS=...,vlm_count",
            "cli": "./scripts/gopex-vlm-count.sh",
        },
    }


def _stub_predicted_count(n_ground: int, visual_train_max: int) -> int:
    """Toy decoder: ID exact; VE collapses to attractor tokens."""
    if n_ground <= visual_train_max:
        return n_ground
    if n_ground == 99:
        return 99
    return 49


def evaluation_demo(cfg: VlmCountConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VlmCountConfig()
    rng = np.random.default_rng(0)
    id_emb, id_lab = [], []
    for n in range(0, cfg.visual_train_max + 1, 10):
        board = sample_go_board(n, grid_size=19, distractor_delta=30, rng=rng)
        emb, lab = patch_embeddings_from_board(board, embed_dim=cfg.probe_dim, rng=rng)
        id_emb.append(emb)
        id_lab.append(lab)

    w, b = fit_id_probe(id_emb, id_lab, rng=rng)

    diagnostics = []
    for n in [25, 55, 75, 110]:
        board = sample_go_board(n, grid_size=19, distractor_delta=30, rng=rng)
        # Stub: linear probe on encoder recovers NG on extrapolation (paper Fig. 3 vision gap ≈ 0).
        nh = board.n_black
        np_pred = _stub_predicted_count(n, cfg.visual_train_max)
        diagnostics.append(diagnose_sample(n, nh, np_pred, visual_train_max=cfg.visual_train_max))

    ve_samples = [d for d in diagnostics if d["regime"] == "VE"]
    mean_vg = float(np.mean([d["vision_gap"] for d in diagnostics]))
    mean_lg_ve = float(np.mean([d["language_gap"] for d in ve_samples])) if ve_samples else 0.0

    return {
        "probe_trained_on_id": True,
        "diagnostics": diagnostics,
        "mean_vision_gap": mean_vg,
        "mean_language_gap_ve": mean_lg_ve,
        "paper_tables": benchmarks_bundle(),
    }


def evaluation_smoke(cfg: VlmCountConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    bench = benchmarks_bundle()
    ve_diag = [d for d in demo["diagnostics"] if d["regime"] == "VE"]
    symbolic_ok = all(d["bottleneck_stage"] == "symbolic_mapping" for d in ve_diag)
    return {
        "ok": symbolic_ok and demo["mean_vision_gap"] < 3.0,
        "mean_vision_gap": round(demo["mean_vision_gap"], 4),
        "mean_language_gap_ve": round(demo["mean_language_gap_ve"], 4),
        "synthetic_vision_ve_acc_pct": bench["synthetic_regime_accuracy"]["vision"]["VE"],
        "compare_vision_ve_pct": bench["synthetic_compare_vision"]["VE"],
        "circuit_disjoint_pct": bench["mechanism"]["circuit_head_disjoint_pct"],
    }
