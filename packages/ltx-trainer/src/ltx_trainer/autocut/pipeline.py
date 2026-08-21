"""AutoCut pipeline cards and demos."""

from __future__ import annotations

from typing import Any

from ltx_trainer.autocut.config import AutoCutConfig
from ltx_trainer.autocut.dataset import alignment_dataset_card, dataset_statistics_summary, sft_dataset_card
from ltx_trainer.autocut.editing import run_footage_driven_edit, run_script_driven_edit
from ltx_trainer.autocut.mock import evaluation_smoke, run_case_demo
from ltx_trainer.autocut.tables import (
    table1_main_results,
    table2_ablation_training,
    table_alignment_sft_hyperparams,
    table_rqvae_hyperparams,
    user_study_vs_gpt4o,
)
from ltx_trainer.autocut.taxonomy import EditingTask, TASK_LABELS


def framework_card(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    return {
        "name": "AutoCut",
        "paper": f"arXiv:{c.paper_arxiv}",
        "github": c.github_url,
        "full_name": "End-to-end advertisement video editing via multimodal discretization",
        "backbone": c.base_llm,
        "encoders": {
            "visual": c.visual_encoder,
            "audio": c.audio_encoder,
        },
        "tokenization": {
            "video": f"{c.video_rqvae.quant_heads}× codebook {c.video_rqvae.codebook_size}",
            "audio": f"{c.audio_rqvae.quant_heads}× codebook {c.audio_rqvae.codebook_size}",
            "target_video_cosine": c.video_rqvae.target_cosine_video,
            "target_audio_cosine": c.audio_rqvae.target_cosine_audio,
        },
        "training": {
            "stage1": "multimodal alignment (frozen LLM, new embeddings)",
            "stage2": "task SFT (full parameters)",
            "alignment_samples": c.alignment_samples,
            "sft_samples": c.sft_samples,
        },
        "tasks": {t.value: TASK_LABELS[t] for t in EditingTask},
        "rendering": "RQ-VAE decode → FAISS NN retrieval → ffmpeg concat + subtitles + TTS + BGM",
        "implementation": "ltx_trainer.autocut.inference.AutoCutEditor",
        "fps": {"reasoning": c.low_fps, "retrieval": c.high_fps},
    }


def knowledge_card(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    ours = [r for r in table1_main_results() if r["model"] == "AutoCut (ours)"][0]
    return {
        "paper_arxiv": c.paper_arxiv,
        "github": c.github_url,
        "insight": "Unified video–audio–text discrete tokens enable controllable ad editing in one LLM",
        "datasets": {
            "alignment": alignment_dataset_card(c),
            "sft": sft_dataset_card(c),
            "statistics": dataset_statistics_summary(),
        },
        "best_results": ours,
        "cost_per_100_videos_usd": {
            "autocut_rtx4090": c.inference_cost_per_100_videos_usd,
            "gpt4o_pipeline": c.gpt4o_cost_per_100_videos_usd,
        },
        "limitations": [
            "Material-based retrieval (no pixel synthesis)",
            "Clip-level control only (not frame-level emotion editing)",
            "Subtle video–audio rhythm desync possible",
        ],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_main": table1_main_results(),
        "table2_ablation": table2_ablation_training(),
        "rqvae_hyperparams": table_rqvae_hyperparams(),
        "training_hyperparams": table_alignment_sft_hyperparams(),
        "user_study_vs_gpt4o": user_study_vs_gpt4o(),
    }


def evaluation_demo(*, case_id: str = "edifier_earbuds", cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    base = run_case_demo(case_id, c)
    base["script_driven"] = run_script_driven_edit(case_id, cfg=c)
    base["footage_driven"] = run_footage_driven_edit(case_id, cfg=c)
    return base


def editing_demo(
    *,
    case_id: str = "edifier_earbuds",
    scenario: str = "script_driven",
    cfg: AutoCutConfig | None = None,
) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    if scenario == "footage_driven":
        return run_footage_driven_edit(case_id, cfg=c)
    return run_script_driven_edit(case_id, cfg=c)
