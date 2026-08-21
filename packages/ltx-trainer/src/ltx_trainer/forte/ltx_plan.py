"""LTX / native-evolve integration for FORTE text–audio alignment (arXiv:2606.05812)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.forte.config import ForteConfig


def ltx_integration_plan(cfg: ForteConfig | None = None) -> dict[str, Any]:
    c = cfg or ForteConfig()
    return {
        "goal": "FOL-refined prompts + T2A alignment sidecars for native LTX AV training",
        "phases": [
            {
                "id": "fold_backfill",
                "action": "GOPEX_ENABLE_AV_FOLD=1 + forte,forte_audio in GOPEX_AV_FOLD_HOOKS",
                "tool": "./scripts/kino-forte-ltx.sh backfill",
                "artifact": "latents/*.pt forte + audio_latents/*.pt forte_audio sidecars",
            },
            {
                "id": "caption_refine",
                "action": "GOPEX_CAPTION_FORTE=1 during process_captions (optional offline q*)",
                "env": "Refines captions before Gemma embed cache write",
            },
            {
                "id": "train_weights",
                "action": "av_fold.use_forte_t2a_weights + use_forte_audio_weights",
                "env": "GOPEX_AV_FOLD_TRAIN=1",
                "effect": "Upweight clips with high t2a_align_proxy / rerank_blend_proxy",
            },
            {
                "id": "infer_refine",
                "action": "GOPEX_INFER_FORTE=1 before live Gemma encode",
                "tool": "./scripts/kino-forte-ltx.sh refine 'birds chirping'",
                "effect": "Stage-1 FOL beam search on user prompt at delivery / inference",
            },
        ],
        "recommended_hooks": [
            "forte",
            "forte_audio",
            "avbench",
            "planaudio",
            "ag_repa",
            "growloop",
        ],
        "use_cases": [
            "T2AV clips where caption–audio predicate overlap matters",
            "pd-horror-movies / gphotos ambient sound alignment",
            "Inference prompt sharpening for foley-heavy prompts",
        ],
        "train_fields": [
            "t2a_align_proxy",
            "caption_fol_overlap",
            "rerank_blend_proxy",
            "fol_predicate_count",
        ],
        "paper": c.paper_arxiv,
    }


def native_evolve_env_snippet() -> str:
    return "\n".join(
        [
            "export GOPEX_ENABLE_AV_FOLD=1",
            "export GOPEX_AV_FOLD_HOOKS=forte,forte_audio,avbench,planaudio,ag_repa",
            "export GOPEX_AV_FOLD_TRAIN=1",
            "export GOPEX_INFER_FORTE=1   # default on; set 0 to skip live q* refinement",
        ]
    )
