"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.aac_audioset.config import AacAudiosetConfig
from ltx_trainer.aac_audioset.fusion import (
    caption_ce_loss,
    decode_step_argmax,
    fuse_acoustic_semantic,
    top_k_audioset_keywords,
)


def framework_card(cfg: AacAudiosetConfig | None = None) -> dict[str, Any]:
    c = cfg or AacAudiosetConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "automated_audio_captioning",
        "components": [
            "ConvNeXt-Tiny audio encoder (AudioSet pretrained)",
            "Frozen ConvNeXt AudioSet top-K keyword classifier",
            "Acoustic-semantic concat H_f = [H_a; K]",
            "Compact 6-layer BART-style decoder (3 enc + 3 dec)",
        ],
        "top_k_keywords": c.top_k_keywords,
        "audioset_classes": c.audioset_class_count,
        "headline": headline_results(c),
    }


def headline_results(cfg: AacAudiosetConfig | None = None) -> dict[str, Any]:
    c = cfg or AacAudiosetConfig()
    return {
        "clotho_in_domain_spider": c.clotho_id_spider,
        "clotho_in_domain_fense": c.clotho_id_fense,
        "audiocaps_in_domain_spider": c.audiocaps_id_spider,
        "audiocaps_in_domain_fense": c.audiocaps_id_fense,
        "beats_kim_clotho_spider": c.clotho_id_spider > c.kim_clotho_id_spider,
        "beats_pengi_audiocaps_spider": c.audiocaps_id_spider > c.pengi_audiocaps_spider,
    }


def table1_clotho() -> list[dict[str, Any]]:
    """Table 1 — evaluation on Clotho."""
    c = AacAudiosetConfig()
    return [
        {"method": "Kim et al. [6]", "setting": "in-domain", "spider": c.kim_clotho_id_spider, "fense": 0.336},
        {"method": "DCASE 2023", "setting": "in-domain", "spider": 0.252, "fense": 0.437},
        {"method": "Ours", "setting": "in-domain", "spider": c.clotho_id_spider, "fense": c.clotho_id_fense,
         "bleu1": c.clotho_id_bleu1, "cider": c.clotho_id_cider},
        {"method": "Kim et al. [6]", "setting": "cross (AudioCaps→Clotho)", "spider": 0.133, "fense": 0.325},
        {"method": "Ours", "setting": "cross (AudioCaps→Clotho)", "spider": c.clotho_xd_spider, "fense": c.clotho_xd_fense},
        {"method": "Pengi [7]", "setting": "large pretrained", "spider": c.pengi_clotho_spider, "fense": 0.488},
        {"method": "Keyword→LLaMA-2-7B", "setting": "prompt-based", "spider": c.llama_clotho_spider, "fense": 0.445},
    ]


def table2_audiocaps() -> list[dict[str, Any]]:
    """Table 2 — evaluation on AudioCaps."""
    c = AacAudiosetConfig()
    return [
        {"method": "Kim et al. [6]", "setting": "in-domain", "spider": 0.455, "fense": 0.357},
        {"method": "Ours", "setting": "in-domain", "spider": c.audiocaps_id_spider, "fense": c.audiocaps_id_fense,
         "bleu1": c.audiocaps_id_bleu1, "cider": c.audiocaps_id_cider},
        {"method": "Kim et al. [6]", "setting": "cross (Clotho→AudioCaps)", "spider": 0.147, "fense": 0.283},
        {"method": "Ours", "setting": "cross (Clotho→AudioCaps)", "spider": c.audiocaps_xd_spider, "fense": c.audiocaps_xd_fense},
        {"method": "Pengi [7]", "setting": "large pretrained", "spider": c.pengi_audiocaps_spider, "fense": 0.494},
        {"method": "Keyword→LLaMA-2-7B", "setting": "prompt-based", "spider": c.llama_audiocaps_spider, "fense": 0.535},
    ]


def table3_keyword_ablation() -> list[dict[str, Any]]:
    """Table 3 — keyword guidance + decoder ablation on Clotho."""
    c = AacAudiosetConfig()
    return [
        {"decoder": "BART-Base", "keyword_guidance": False, "params_m": 169, "spider": 0.248},
        {"decoder": "BART-Large", "keyword_guidance": False, "params_m": 437, "spider": 0.179},
        {"decoder": "Ours", "keyword_guidance": False, "params_m": c.params_m, "spider": c.ours_no_kw_spider},
        {"decoder": "BART-Base", "keyword_guidance": True, "params_m": 169, "spider": 0.267},
        {"decoder": "BART-Large", "keyword_guidance": True, "params_m": 437, "spider": 0.241},
        {"decoder": "Ours", "keyword_guidance": True, "params_m": c.params_m, "spider": c.ours_kw_spider,
         "bleu1": c.ours_kw_bleu1},
    ]


def table4_k_sensitivity() -> list[dict[str, Any]]:
    """Table 4 — effect of K on Clotho."""
    c = AacAudiosetConfig()
    return [
        {"k": 5, "bleu1": 0.602, "meteor": 0.182, "rouge_l": 0.394, "cider": 0.446, "spider": c.k5_spider},
        {"k": 10, "bleu1": 0.598, "meteor": 0.183, "rouge_l": 0.392, "cider": 0.437, "spider": c.k10_spider},
        {"k": 15, "bleu1": 0.598, "meteor": 0.180, "rouge_l": 0.392, "cider": 0.441, "spider": c.k15_spider},
    ]


def benchmarks_bundle(cfg: AacAudiosetConfig | None = None) -> dict[str, Any]:
    c = cfg or AacAudiosetConfig()
    return {
        "table1_clotho": table1_clotho(),
        "table2_audiocaps": table2_audiocaps(),
        "table3_keyword_ablation": table3_keyword_ablation(),
        "table4_k_sensitivity": table4_k_sensitivity(),
        "optimal_k": c.top_k_keywords,
    }


def evaluation_demo(seed: int = 42, cfg: AacAudiosetConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or AacAudiosetConfig()

    t_frames, m_tokens = 32, 12
    acoustic = rng.normal(0, 1, (t_frames, c.model_dim))
    logits = rng.normal(0, 1, c.audioset_class_count)
    top_k = top_k_audioset_keywords(logits, k=c.top_k_keywords)
    keyword_emb = rng.normal(0, 1, (m_tokens, c.model_dim))
    fused = fuse_acoustic_semantic(acoustic, keyword_emb)

    log_probs = rng.random((4, 100))
    log_probs /= log_probs.sum(axis=-1, keepdims=True)
    targets = np.zeros_like(log_probs)
    targets[np.arange(4), rng.integers(0, 100, 4)] = 1.0
    loss = caption_ce_loss(log_probs, targets)
    token = decode_step_argmax(rng.normal(0, 1, 100))

    ablation = table3_keyword_ablation()
    ours_kw = next(r for r in ablation if r["decoder"] == "Ours" and r["keyword_guidance"])
    ours_no = next(r for r in ablation if r["decoder"] == "Ours" and not r["keyword_guidance"])

    return {
        "top_k_class_ids": top_k,
        "fused_shape": list(fused.shape),
        "caption_ce_loss": loss,
        "decode_token_id": token,
        "keyword_guidance_improves_spider": ours_kw["spider"] > ours_no["spider"],
        "optimal_k": c.top_k_keywords,
        "beats_llama_keywords_only": c.clotho_id_spider > c.llama_clotho_spider,
    }


def pipeline_demo(seed: int = 42, cfg: AacAudiosetConfig | None = None) -> dict[str, Any]:
    c = cfg or AacAudiosetConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
