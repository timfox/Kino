"""LALM post-training as implicit CL — §4, Fig. 1."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ltx_trainer.speechcl.mitigation import MitigationMechanism


class LalmStage(str, Enum):
    TEXT_PRETRAIN = "stage_1_text_llm_pretraining"
    SPEECH_ALIGNMENT = "stage_2_speech_encoder_alignment"
    INSTRUCTION_TUNING = "stage_3_multi_task_instruction_tuning"
    PREFERENCE_ALIGNMENT = "stage_4_rlhf_preference_alignment"


def stage_transitions() -> list[dict[str, Any]]:
    """Four-stage pipeline with risks and mitigations (Fig. 1)."""
    return [
        {
            "from": LalmStage.TEXT_PRETRAIN.value,
            "to": LalmStage.SPEECH_ALIGNMENT.value,
            "at_risk": ["text reasoning", "world knowledge"],
            "mitigations": [
                MitigationMechanism.ARCHITECTURAL_ISOLATION.value,
            ],
            "practice": "Freeze text backbone; train speech encoder",
            "refs": ["Hsiao et al., 2025", "Cuervo et al., 2025"],
        },
        {
            "from": LalmStage.SPEECH_ALIGNMENT.value,
            "to": LalmStage.INSTRUCTION_TUNING.value,
            "at_risk": ["speech-text alignment", "acoustic representations"],
            "mitigations": [
                MitigationMechanism.REPLAY.value,
                MitigationMechanism.ARCHITECTURAL_ISOLATION.value,
            ],
            "practice": "Mix text+speech instructions; LoRA/adapters",
            "refs": ["Chu et al., 2024", "Xu et al., 2024", "Liu et al., 2024"],
        },
        {
            "from": LalmStage.INSTRUCTION_TUNING.value,
            "to": LalmStage.PREFERENCE_ALIGNMENT.value,
            "at_risk": [
                "instruction following",
                "legacy speech tasks (ASR, TTS)",
            ],
            "mitigations": [
                MitigationMechanism.REGULARIZATION.value,
                MitigationMechanism.REPLAY.value,
            ],
            "practice": "Cross-modal distillation + SFT replay; on-policy RL (KL-minimal)",
            "refs": ["Wang et al., 2025a", "Shenfeld et al."],
        },
    ]


def hybrid_cl_consensus() -> dict[str, Any]:
    return {
        "observation": "LALM post-training relies on hybrid CL by necessity",
        "typical_combo": [
            "frozen text backbone",
            "text data replay",
            "cross-modal distillation",
        ],
        "gap": "Formed in practice without complete theoretical explanation",
    }
