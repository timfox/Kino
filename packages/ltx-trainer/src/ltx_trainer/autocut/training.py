"""Training stage configs (alignment + SFT, Supp. Tables 3–4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.autocut.config import AutoCutConfig
from ltx_trainer.autocut.tables import table_alignment_sft_hyperparams, table_rqvae_hyperparams


def rqvae_training_config(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    rows = table_rqvae_hyperparams()
    by_mod = {r["modality"]: r for r in rows}
    return {
        "video": by_mod["video"],
        "audio": by_mod["audio"],
        "objective": "cosine reconstruction (Eq. 1)",
        "monitor": "cosine_sim",
        "targets": {
            "video": c.video_rqvae.target_cosine_video,
            "audio": c.audio_rqvae.target_cosine_audio,
        },
    }


def alignment_training_config(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    hp = {r["stage"]: r for r in table_alignment_sft_hyperparams()}["alignment"]
    return {
        "stage": "multimodal_alignment",
        "base_model": c.base_llm,
        "frozen_backbone": True,
        "trainable": ["multimodal_embedding_layers"],
        "loss": "next_token_prediction (Eq. 2)",
        "samples": c.alignment_samples,
        "cutoff": c.alignment_cutoff,
        "hyperparams": hp,
    }


def sft_training_config(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    hp = {r["stage"]: r for r in table_alignment_sft_hyperparams()}["sft"]
    return {
        "stage": "supervised_finetuning",
        "base_model": "aligned_autocut",
        "frozen_backbone": False,
        "loss_mask": "response_tokens_only",
        "samples": c.sft_samples,
        "cutoff": c.sft_cutoff,
        "tasks": ["video_selection", "video_sorting", "script_generation", "bgm_selection"],
        "hyperparams": hp,
    }


def full_training_pipeline(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    return {
        "rqvae": rqvae_training_config(cfg),
        "alignment": alignment_training_config(cfg),
        "sft": sft_training_config(cfg),
        "recommended": "emb+sft (two-stage; avoid noisy emb+full+sft per Table 2)",
    }
