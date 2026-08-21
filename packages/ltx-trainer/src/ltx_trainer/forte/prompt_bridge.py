"""FORTE Stage-1 prompt refinement for LTX / Gemma text encode (arXiv:2606.05812)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.forte.config import ForteConfig
from ltx_trainer.forte.infer import caption_forte_enabled, infer_forte_beam_online, infer_forte_enabled
from ltx_trainer.forte.search import RefinementResult, SearchConfig, best_first_search


def _search_config(cfg: ForteConfig | None = None) -> SearchConfig:
    c = cfg or ForteConfig()
    b, d = c.beam_online if infer_forte_beam_online() else c.beam_offline
    return SearchConfig(
        beam_width=b,
        max_depth=d,
        lambda_neg=c.lambda_neg,
        beta_pivot=c.beta_pivot,
        tau=c.feasibility_tau,
        embed_dim=c.embed_dim,
    )


def refine_prompt_for_ltx(
    prompt: str,
    *,
    cfg: ForteConfig | None = None,
    anchor_caption: str | None = None,
) -> dict[str, Any]:
    """Run FORTE Stage-1 beam search; return q* and metadata for logging / sidecars."""
    text = (prompt or "").strip()
    if not text:
        return {"q_original": prompt, "q_star": prompt, "score": 0.0, "explored": 0, "refined": False}
    result: RefinementResult = best_first_search(text, cfg=_search_config(cfg))
    out: dict[str, Any] = {
        "q_original": text,
        "q_star": result.q_star,
        "score": round(float(result.score), 4),
        "explored": int(result.explored),
        "phi_star": result.phi_star.raw,
        "refined": result.q_star.strip() != text,
    }
    if anchor_caption:
        from ltx_trainer.forte.fol import parse_query_fallback, pred_set, predicate_overlap
        from ltx_trainer.forte.rerank import caption_to_fol

        phi_q = result.phi_star
        phi_c = caption_to_fol(anchor_caption) if anchor_caption.strip() else parse_query_fallback(anchor_caption)
        out["caption_fol_overlap"] = round(predicate_overlap(phi_q, phi_c), 4)
        out["fol_predicate_count"] = len(pred_set(phi_q))
    return out


def maybe_refine_ltx_prompt(prompt: str, *, anchor_caption: str | None = None) -> str:
    """Return q* when inference/caption FORTE gates are on; else unchanged prompt."""
    if not (infer_forte_enabled() or caption_forte_enabled()):
        return prompt
    return str(refine_prompt_for_ltx(prompt, anchor_caption=anchor_caption)["q_star"])


def maybe_refine_prompts(prompts: list[str]) -> list[str]:
    """Refine the first prompt when ``GOPEX_INFER_FORTE=1`` (T2V / T2AV live encode)."""
    if not infer_forte_enabled() or not prompts:
        return prompts
    out = list(prompts)
    out[0] = maybe_refine_ltx_prompt(out[0])
    return out
