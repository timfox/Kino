"""OmniCustom pipeline cards and demo entrypoints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.omnicustom.benchmark import BENCHMARK_CASES, case_by_id
from ltx_trainer.omnicustom.config import OmniCustomConfig
from ltx_trainer.omnicustom.dataset import dataset_card
from ltx_trainer.omnicustom.mock import evaluation_smoke
from ltx_trainer.omnicustom.prompts import extract_speech, strip_speech_tags
from ltx_trainer.omnicustom.tables import appendix_lse_comparison, table2_quantitative, table3_user_study
from ltx_trainer.omnicustom.taxonomy import CustomizationSetting, SETTING_TRAITS


def framework_card(cfg: OmniCustomConfig | None = None) -> dict[str, Any]:
    c = cfg or OmniCustomConfig()
    return {
        "name": "OmniCustom",
        "paper": f"arXiv:{c.paper_arxiv}",
        "task": "sync audio-video customization",
        "inputs": ["reference_image", "reference_audio", "text_prompt_with_<S>/<E>"],
        "outputs": ["identity_preserving_video", "timbre_cloned_audio", "background_sfx"],
        "backbone": c.base_model,
        "modules": [
            "reference_image_branch + identity LoRA (video self-attention QKV)",
            "reference_audio_branch + timbre LoRA (audio self-attention QKV)",
            "InsightFace + NaturalSpeech3 embedding injection",
            "contrastive flow regularization (identity + timbre)",
        ],
        "loss": {
            "lambda_video_fm": c.lambda_video_fm,
            "lambda_audio_fm": c.lambda_audio_fm,
            "lambda_identity_cl": c.lambda_identity_cl,
            "lambda_timbre_cl": c.lambda_timbre_cl,
        },
        "inference": {
            "flow_steps": c.flow_steps,
            "guidance_video": c.guidance_video,
            "guidance_audio": c.guidance_audio,
        },
    }


def knowledge_card(cfg: OmniCustomConfig | None = None) -> dict[str, Any]:
    c = cfg or OmniCustomConfig()
    return {
        "paper_arxiv": c.paper_arxiv,
        "project_url": c.project_url,
        "novel_task": "First sync audio-video customization: joint identity + timbre in one pass",
        "vs_audio_driven": "No TTS pre-step; speech content from text; background SFX from joint AV model",
        "dataset": dataset_card(c),
        "benchmark_size": c.benchmark_size,
        "limitations": [
            "English only (base OVI constraint)",
            "5 second clips",
            "weak profile-face identity preservation",
        ],
        "settings": {k.value: v for k, v in SETTING_TRAITS.items()},
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table2": table2_quantitative(),
        "table3_user_study": table3_user_study(),
        "appendix_lse": appendix_lse_comparison(),
        "mini_cases": [{"id": x.case_id, "prompt": x.prompt} for x in BENCHMARK_CASES],
    }


def evaluation_demo(*, case_id: str = "sydney_harbour", cfg: OmniCustomConfig | None = None) -> dict[str, Any]:
    c = cfg or OmniCustomConfig()
    case = case_by_id(case_id) or BENCHMARK_CASES[0]
    parsed = extract_speech(case.prompt, c)
    baseline_prompt = strip_speech_tags(case.prompt)
    smoke = evaluation_smoke(c)
    return {
        "case_id": case.case_id,
        "prompt": case.prompt,
        "scene_only": parsed.scene_text,
        "speech": parsed.speech_text,
        "baseline_prompt_no_tags": baseline_prompt,
        "setting": CustomizationSetting.SYNC_AV.value,
        "smoke_summary": {
            k: smoke[k]
            for k in (
                "loss",
                "stub_facesim_arc",
                "paper_facesim_arc",
                "paper_fvd",
                "paper_speaker_sim",
            )
        },
    }
