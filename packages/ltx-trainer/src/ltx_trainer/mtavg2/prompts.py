"""Prompt stubs from MTAVG-Bench 2.0 appendix (construction + evaluation)."""

from __future__ import annotations

SHOT_ANALYSIS_SYSTEM_PROMPT_EXCERPT = """\
You are an expert film analyst and content transformer. Analyze the uploaded film clip and \
convert it into a copyright-safe, neutral, structured JSON plan for chained video generation. \
Split the clip into sequential 8.0-second generation segments with exact dialogue fidelity. \
Output JSON only with clip_summary, model_inference_rules, and segments."""

VIDEO_GENERATION_SYSTEM_PROMPT_EXCERPT = """\
You are a cinematic multi-talker segment expert. Generate one multi-shot video segment from \
script metadata: premise, character registry, dialogue (exact lines), emotional tone, and \
continuity. Realize multi-shot cinematic structure with sequential shot changes; single full \
frame only (no split screen). Infer lighting, music, performance, and camera from dramatic context."""

RC_JUDGE_PROMPT_TEMPLATE = """\
You are a rigorous reasoning evaluation expert. Evaluate whether the model's rationale \
logically supports its final answer choice.
[Question]
{question}
[Options]
{options}
[Model's Predicted Option]
{predicted_option_id}: {predicted_option_text}
[Model's Rationale]
{rationale}
Evaluation Criteria (Likert Scale 1-5):
- 1: Rationale completely contradicts or is irrelevant to the chosen option
- 5: Rationale fully and logically supports the chosen option
Output Format (JSON):
{{"score": <integer 1-5>, "rationale": "<detailed analysis>"}}"""

DIAGNOSTIC_QA_INSTRUCTION = """\
Given the multi-talker audio-video clip and scene script context, identify the dominant \
high-level cinematic failure mode under sub-dimension {sub_dim} ({sub_dim_name}). \
Choose from the provided failure-mode options. Ground your answer in visible acting, \
atmosphere, or shot-organization evidence—not only lip-sync or low-level alignment."""

LTX_SCRIPT_PROMPT_SUFFIX = """\
Scene-level cinematic requirements (MTAVG-Bench 2.0): maintain believable character performance, \
coherent mood and soundscape, and motivated shot progression with continuity across cuts. \
Multi-talker dialogue must preserve speaker identity, turn-taking, eyelines, and emotional tone."""


def format_rc_judge_prompt(
    *,
    question: str,
    options: str,
    predicted_option_id: str,
    predicted_option_text: str,
    rationale: str,
) -> str:
    return RC_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        options=options,
        predicted_option_id=predicted_option_id,
        predicted_option_text=predicted_option_text,
        rationale=rationale,
    )


def format_diagnostic_instruction(*, sub_dim: str, sub_dim_name: str) -> str:
    return DIAGNOSTIC_QA_INSTRUCTION.format(sub_dim=sub_dim, sub_dim_name=sub_dim_name)
