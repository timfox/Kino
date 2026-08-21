"""AutoCut editing tasks and pipeline stages."""

from __future__ import annotations

from enum import Enum


class EditingTask(str, Enum):
    VIDEO_SELECTION = "video_selection"
    VIDEO_SORTING = "video_sorting"
    SCRIPT_GENERATION = "script_generation"
    BGM_SELECTION = "bgm_selection"


class TrainingStage(str, Enum):
    MULTIMODAL_ALIGNMENT = "multimodal_alignment"
    SUPERVISED_FINETUNING = "supervised_finetuning"


class RenderStrategy(str, Enum):
    BY_FRAME = "by_frame"
    BY_CLIP = "by_clip"


TASK_LABELS: dict[EditingTask, str] = {
    EditingTask.VIDEO_SELECTION: "Select relevant clips from candidate pool (CSA)",
    EditingTask.VIDEO_SORTING: "Order clips to match script narrative (CRA)",
    EditingTask.SCRIPT_GENERATION: "Generate sentence-level ad script (SQ, WCD)",
    EditingTask.BGM_SELECTION: "Retrieve BGM via discrete audio tokens (MSS)",
}
