"""UNIVID caption, moderation, and CapBench evaluation prompts (arXiv:2606.05748)."""

from __future__ import annotations

CAPTION_PROMPT = """Given image frames uniformly sampled from a video clip, describe the video (not the individual images) in detail, focusing on the main subjects, their actions, and the background scene. You should also pay attention to details that may violate the trust-and-safety platform content policy. Don't describe feelings or atmosphere.
Your output should be a single coherent paragraph. Maximum length: 150 words."""

CAPTION_INSTRUCTION = """Describe the video briefly, and give possible violation policy."""

LITE_INFERENCE_TEMPLATE = """Region: {region}
Title: {title}
User nickname: {nickname}
ASR Text: {asr}
OCR Text: {ocr}
Bio Text: {profile}
Based on the video frames and related content above, please indicate the severity of any inappropriate, disruptive, or harmful content."""

RAG_INFERENCE_TEMPLATE = """Region: {region}
Title: {title}
User nickname: {nickname}
ASR Text: {asr}
OCR Text: {ocr}
Bio Text: {profile}
Violative Events for Reference:
{violative_events}
Based on the video frames, the above content, and the judgment logic of the similar cases, please indicate the severity of any inappropriate, disruptive, or harmful content."""

EVENT_EXTRACTION_TEMPLATE = """You are given a description of a video clip: {caption}
Your tasks:
1. Extract at most 10 key events from the description.
2. From these events, identify those that may violate the video platform content policy labels: {policy_list}.
Only include events that can serve as sufficient evidence of the listed policy violations.
Output Format: Return a Python dictionary string with keys "events" and "violative_events"."""

ENTAILMENT_JUDGMENT_TEMPLATE = """You are given a video description and a list of events.
Classify the relationship between the video description and each event as entailment, contradiction, or neutral.
Video Description: {caption}
Events: {events}
Return JSON list with event and relationship fields only."""

VQA_TASKS: tuple[str, ...] = ("caption", "summary", "topic", "keywords")


def format_lite_prompt(
    *,
    region: str = "global",
    title: str = "",
    nickname: str = "",
    asr: str = "",
    ocr: str = "",
    profile: str = "",
) -> str:
    return LITE_INFERENCE_TEMPLATE.format(
        region=region,
        title=title or "(none)",
        nickname=nickname or "(none)",
        asr=asr or "(none)",
        ocr=ocr or "(none)",
        profile=profile or "(none)",
    )


def format_rag_prompt(
    retrieved: list[tuple[str, str]],
    *,
    region: str = "global",
    title: str = "",
    nickname: str = "",
    asr: str = "",
    ocr: str = "",
    profile: str = "",
) -> str:
    lines = []
    for i, (policy, reason) in enumerate(retrieved, start=1):
        lines.append(f"{i}. {policy}, violation reason: {reason}")
    events_block = "\n".join(lines) if lines else "(none)"
    return RAG_INFERENCE_TEMPLATE.format(
        region=region,
        title=title or "(none)",
        nickname=nickname or "(none)",
        asr=asr or "(none)",
        ocr=ocr or "(none)",
        profile=profile or "(none)",
        violative_events=events_block,
    )


def prompt_bundle() -> dict[str, object]:
    return {
        "caption_max_words": 150,
        "vqa_tasks": list(VQA_TASKS),
        "rag_top_k_default": 3,
        "lite_fields": ["region", "title", "nickname", "asr", "ocr", "profile"],
    }
