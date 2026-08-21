"""Knowledge cards, training plan, and evaluation demo."""

from __future__ import annotations

from typing import Any, Literal

from ltx_trainer.id_lora.benchmarks import summary_anchors
from ltx_trainer.id_lora.config import IdLoraConfig
from ltx_trainer.id_lora.inference_plan import build_inference_argv, inference_defaults
from ltx_trainer.id_lora.prompts import build_structured_prompt, parse_id_lora_prompt


def framework_card(cfg: IdLoraConfig | None = None) -> dict[str, Any]:
    c = cfg or IdLoraConfig()
    return {
        "arxiv": c.paper_arxiv,
        "title": c.paper_title,
        "url": c.project_url,
        "method": "In-context LoRA on LTX-2/2.3: reference audio + first-frame video + structured prompt",
        "training_strategy": "audio_ref_only_ic",
        "key_mechanisms": [
            "Negative temporal RoPE positions for reference audio",
            "Identity guidance on audio during inference",
            "Rank-128 LoRA on audio + cross-modal layers",
            "Mask ref audio from cross-modal and text attention (training)",
        ],
        "prompt_tags": ["[VISUAL]", "[SPEECH]", "[SOUNDS]"],
        "inference_modes": ["one_stage", "two_stage", "two_stage_hq (LTX-2.3)"],
    }


def knowledge_card(cfg: IdLoraConfig | None = None) -> dict[str, Any]:
    c = cfg or IdLoraConfig()
    return {
        "framework": framework_card(c),
        "datasets": dict(c.hf_datasets),
        "checkpoints": dict(c.hf_checkpoints),
        "training": training_plan(c),
        "inference_defaults": inference_defaults(c),
        "summary": summary_anchors(),
    }


def training_plan(cfg: IdLoraConfig | None = None) -> dict[str, Any]:
    c = cfg or IdLoraConfig()
    return {
        "strategy": "audio_ref_only_ic",
        "lora_rank": c.lora_rank,
        "use_negative_ref_positions": c.use_negative_ref_positions,
        "mask_cross_attention_to_reference": c.mask_cross_attention_to_reference,
        "mask_reference_from_text_attention": c.mask_reference_from_text_attention,
        "preprocess": [
            "video latents",
            "audio latents (target)",
            "reference_audio_latents",
            "conditions (Gemma text)",
        ],
        "gopex_config": "configs/ltx2_id_lora_audio_ref_ic.yaml",
        "upstream_configs": [
            "ID-LoRA/configs/training_celebvhq.yaml",
            "ID-LoRA/configs/training_talkvid.yaml",
        ],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return summary_anchors()


def evaluation_demo(
    *,
    speech: str = "We are proud to introduce ID-LoRA.",
    visual: str | None = None,
    mode: Literal["one_stage", "two_stage", "two_stage_hq"] = "one_stage",
) -> dict[str, Any]:
    c = IdLoraConfig()
    visual = visual or (
        "A medium shot of a person speaking naturally indoors, soft warm lighting, shallow depth of field."
    )
    prompt = build_structured_prompt(visual, speech, sounds="Conversational tone, close mic, subtle room tone.")
    parsed = parse_id_lora_prompt(prompt)
    plan = build_inference_argv(mode=mode, prompt=prompt)
    return {
        "prompt": prompt,
        "parsed": {
            "visual_chars": len(parsed.visual),
            "speech_chars": len(parsed.speech),
            "identity_ready": parsed.identity_ready,
        },
        "inference_plan": plan,
        "training_strategy": training_plan(c),
    }
