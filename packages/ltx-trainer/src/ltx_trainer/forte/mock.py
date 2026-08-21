"""Toy FORTE text-to-audio retrieval smoke (arXiv:2606.05812)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.forte.align import ProjectionMLP, train_step_stub
from ltx_trainer.forte.fol import parse_query_fallback, verbalise
from ltx_trainer.forte.metrics import metrics_from_ranking
from ltx_trainer.forte.rerank import rerank
from ltx_trainer.forte.search import SearchConfig, _hash_embed, best_first_search


def run_fol_refinement_smoke(*, seed: int = 0) -> dict[str, Any]:
    query = "birds chirping in the morning"
    result = best_first_search(query, cfg=SearchConfig(beam_width=3, max_depth=3))
    return {
        "query": query,
        "phi0_preds": sorted(result.phi0.predicates),
        "phi_star_preds": sorted(result.phi_star.predicates),
        "phi_star_neg": sorted(result.phi_star.negated),
        "q_star": result.q_star,
        "explored": result.explored,
        "score": round(result.score, 4),
    }


def run_alignment_smoke(*, seed: int = 0) -> dict[str, Any]:
    module = ProjectionMLP(dim=512)
    step = train_step_stub(module, batch_size=8)
    return {"params_m": round(step["params_m"], 3), "loss": round(step["loss"], 4), "gamma": step["gamma"]}


def run_rerank_smoke(*, seed: int = 0) -> dict[str, Any]:
    query = "quiet footsteps in an empty corridor"
    ref = best_first_search(query, cfg=SearchConfig(beam_width=3, max_depth=3))
    q_emb = _hash_embed(ref.q_star, 512)
    module = ProjectionMLP(dim=512)
    candidates = [
        ("a1", "quiet indoor footsteps in empty hallway", _hash_embed("quiet indoor footsteps", 512)),
        ("a2", "loud crowd running outdoors", _hash_embed("loud crowd running", 512)),
        ("a3", "soft footsteps on carpet in empty room", _hash_embed("soft footsteps carpet", 512)),
        ("a4", "heavy boots on gravel path", _hash_embed("heavy boots gravel", 512)),
    ]
    ranked = rerank(q_emb, candidates, ref.phi_star, alpha=0.3, project_fn=module.forward)
    top_id = ranked[0].audio_id
    relevant = {"a1", "a3"}
    m = metrics_from_ranking(relevant, [r.audio_id for r in ranked])
    return {
        "query": query,
        "top_id": top_id,
        "top_score": round(ranked[0].score, 4),
        "r_at_1": round(m.r_at_1, 3),
        "map_at_10": round(m.map_at_10, 3),
    }


def run_pipeline_smoke(*, seed: int = 0) -> dict[str, Any]:
    """End-to-end: refine → project → rerank on speaking query."""
    query = "a person talking"
    ref = best_first_search(query)
    q_emb = _hash_embed(ref.q_star, 512)
    module = ProjectionMLP()
    pool = [
        ("s1", "person speaking quietly indoors", _hash_embed("quiet speaking", 512)),
        ("s2", "person shouting angrily", _hash_embed("shouting angry", 512)),
        ("s3", "crowd vocalisation at stadium", _hash_embed("crowd vocalisation", 512)),
        ("s4", "calm conversation between two people", _hash_embed("calm conversation", 512)),
    ]
    ranked = rerank(q_emb, pool, ref.phi_star, project_fn=module.forward)
    return {
        "query": query,
        "q_star": ref.q_star,
        "has_negation": bool(ref.phi_star.negated),
        "top_caption": ranked[0].caption,
        "top_id": ranked[0].audio_id,
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    fol = run_fol_refinement_smoke(seed=seed)
    align = run_alignment_smoke(seed=seed + 1)
    rer = run_rerank_smoke(seed=seed + 2)
    pipe = run_pipeline_smoke(seed=seed + 3)
    return {
        "fol_refinement": fol,
        "alignment": align,
        "rerank": rer,
        "pipeline": pipe,
        "pipeline_ok": rer["r_at_1"] >= 0.5 and pipe["top_id"] in ("s1", "s4") and len(fol["phi_star_preds"]) >= len(fol["phi0_preds"]),
    }
