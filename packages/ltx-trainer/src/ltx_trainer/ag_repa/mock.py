"""AG-REPA evaluation smoke (arXiv:2603.01006)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ag_repa.ag_repa import total_ag_repa_loss
from ltx_trainer.ag_repa.config import AgRepaConfig
from ltx_trainer.ag_repa.dataset import dataset_card
from ltx_trainer.ag_repa.fog_a import top_k_by_fog_a
from ltx_trainer.ag_repa.lasp import top_k_layers
from ltx_trainer.ag_repa.profiles import synthetic_fog_profile, synthetic_lasp_profile
from ltx_trainer.ag_repa.tables import (
    table1_scd_dissociation,
    table2_alignment_strategies,
    table3_selection_targets,
    table6_efficiency,
)
from ltx_trainer.ag_repa.taxonomy import SelectionStrategy, STRATEGY_LABELS


def evaluation_smoke(cfg: AgRepaConfig | None = None) -> dict[str, Any]:
    c = cfg or AgRepaConfig()
    rng = np.random.default_rng(c.random_seed)

    fog = synthetic_fog_profile(c, seed=c.random_seed)
    lasp_sem = synthetic_lasp_profile(c, domain="semantic")
    lasp_evt = synthetic_lasp_profile(c, domain="event")

    fog_top = top_k_by_fog_a(fog, c.top_k_layers)
    lasp_top = top_k_layers(lasp_sem, c.top_k_layers)

    d = 32
    v_pred = rng.normal(size=(8, d))
    v_tgt = v_pred + rng.normal(scale=0.05, size=v_pred.shape)
    speech_p = rng.normal(size=d)
    speech_t = speech_p + rng.normal(scale=0.1, size=d)
    audio_p = rng.normal(size=d)
    audio_t = audio_p + rng.normal(scale=0.1, size=d)

    layer_proj = {l: rng.normal(size=d) for l in fog_top}
    layer_teach = {l: layer_proj[l] + rng.normal(scale=0.08, size=d) for l in fog_top}

    loss = total_ag_repa_loss(
        v_pred=v_pred,
        v_target=v_tgt,
        speech_proj=speech_p,
        speech_teacher=speech_t,
        audio_proj=audio_p,
        audio_teacher=audio_t,
        layer_projections=layer_proj,
        layer_teachers=layer_teach,
        fog_scores=fog,
        cfg=c,
    )

    t2_best = [r for r in table2_alignment_strategies() if r["method"] == "AG-REPA (Top-3)"][0]
    t3_best = [r for r in table3_selection_targets() if "FoG-A" in r["strategy"]][0]
    scd_b = table1_scd_dissociation()[1]

    overlap_fog_lasp = len(set(fog_top) & set(lasp_top))

    return {
        "paper": f"arXiv:{c.paper_arxiv}",
        "github": c.github_url,
        "phenomenon": "Store-Contribute Dissociation (SCD)",
        "diagnostics": ["BiT-C", "LASP", "FoG-A"],
        "selection_strategy": SelectionStrategy.AG_REPA.value,
        "fog_top3_speech_paper": list(c.paper_top3_speech_fog),
        "fog_top3_synthetic": fog_top,
        "lasp_top3_synthetic": lasp_top,
        "scd_overlap_top3": overlap_fog_lasp,
        "config_b_fog_speech_top3": scd_b["fog_speech_top3"],
        "loss": loss.to_dict(),
        "paper_speech_fad": t2_best["speech_fad"],
        "paper_audio_fad": t2_best["audio_fad"],
        "paper_speech_wer": t2_best["speech_wer"],
        "paper_fog_fad_gain_pct": t3_best["rel_gain_pct"],
        "paper_convergence_steps": t3_best["convergence_steps"],
        "efficiency_speedup_vs_lasp": "≈3.26×",
        "efficiency_table6": table6_efficiency(),
        "dataset": dataset_card(c),
        "strategy_labels": STRATEGY_LABELS[SelectionStrategy.AG_REPA],
    }
