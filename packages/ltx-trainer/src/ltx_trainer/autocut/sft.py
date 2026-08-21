"""Task-oriented SFT formatting (ShareGPT-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.autocut.taxonomy import EditingTask


@dataclass
class SFTExample:
    task: EditingTask
    system: str
    human: str
    assistant: str

    def to_sharegpt(self) -> dict[str, Any]:
        return {
            "conversations": [
                {"from": "system", "value": self.system},
                {"from": "human", "value": self.human},
                {"from": "gpt", "value": self.assistant},
            ],
            "task": self.task.value,
        }


def _system_prompt(task: EditingTask) -> str:
    if task == EditingTask.VIDEO_SELECTION:
        return "You are a professional video editor. Select clips relevant to the product and script."
    if task == EditingTask.VIDEO_SORTING:
        return "You are a professional video editor. Sort clips by script order."
    if task == EditingTask.SCRIPT_GENERATION:
        return "You are a script writer. Generate a script based on the video content."
    return "You are a music supervisor. Select a matching BGM."


def build_video_selection_example(
    *,
    product_info: str,
    script: str,
    candidate_indices: list[int],
    selected_indices: list[int],
) -> SFTExample:
    human = (
        f"Product:\n{product_info}\n\nScript:\n{script}\n\n"
        f"Candidates (indices): {candidate_indices}\n"
        "Return indices of clips to include."
    )
    return SFTExample(
        task=EditingTask.VIDEO_SELECTION,
        system=_system_prompt(EditingTask.VIDEO_SELECTION),
        human=human,
        assistant=str(selected_indices),
    )


def build_video_sorting_example(
    *,
    product_info: str,
    script: str,
    shuffled_indices: list[int],
    sorted_indices: list[int],
) -> SFTExample:
    human = (
        f"Product:\n{product_info}\n\nScript:\n{script}\n\n"
        f"Clips (shuffled indices): {shuffled_indices}"
    )
    return SFTExample(
        task=EditingTask.VIDEO_SORTING,
        system=_system_prompt(EditingTask.VIDEO_SORTING),
        human=human,
        assistant=str(sorted_indices),
    )


def build_script_generation_example(
    *,
    product_info: str,
    clip_count: int,
    script_lines: list[str],
) -> SFTExample:
    human = f"Product:\n{product_info}\n\nClips: {clip_count} ordered segments."
    return SFTExample(
        task=EditingTask.SCRIPT_GENERATION,
        system=_system_prompt(EditingTask.SCRIPT_GENERATION),
        human=human,
        assistant="\n".join(script_lines),
    )


def build_bgm_selection_example(
    *,
    product_info: str,
    script: str,
    audio_token_line: str,
) -> SFTExample:
    human = f"Product:\n{product_info}\n\nScript:\n{script}\n\nSelect BGM tokens."
    return SFTExample(
        task=EditingTask.BGM_SELECTION,
        system=_system_prompt(EditingTask.BGM_SELECTION),
        human=human,
        assistant=audio_token_line,
    )


def sft_loss_mask(response_tokens: int, total_tokens: int) -> float:
    """Fraction of sequence used for SFT loss (response only)."""
    if total_tokens <= 0:
        return 0.0
    return min(1.0, response_tokens / total_tokens)
