"""InstructAV2AV — instruction-guided audio-video joint editing (arXiv:2605.18467)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InstructAV2AVConfig:
    """Defaults from Sec. 4–5 and InsAVE-80K (Sec. 3)."""

    name: str = "InstructAV2AV"
    paper_arxiv: str = "arXiv:2605.18467"
    website: str = "https://hjzheng.net/projects/InstructAV2AV/"
    title: str = "Instruction-Guided Audio-Video Joint Editing"

    # Backbone (Ovi-style dual stream, Sec. 4.1)
    base_model: str = "Ovi twin-backbone (pretrained AV generation)"
    text_encoder: str = "T5"
    resolution: str = "720p"
    fps: int = 24
    audio_sample_rate_hz: int = 16000
    clip_duration_s: float = 5.0

    # InsAVE-80K (Sec. 3.4)
    dataset_name: str = "InsAVE-80K"
    train_pairs: int = 79_000
    eval_pairs: int = 1_000

    # Flow matching weights Eq. (3)
    lambda_video: float = 0.85
    lambda_audio: float = 0.15

    # Training (Sec. 5.1)
    lr: float = 1e-5
    optimizer: str = "AdamW"
    stage1_video_steps: int = 320_000
    stage1_audio_steps: int = 960_000
    stage2_joint_steps: int = 400_000
    gpus: int = 8
    gpu_type: str = "H100"

    # Data verification criteria (Sec. 3.3)
    verification_criteria: tuple[str, ...] = (
        "instruction_fidelity",
        "content_preservation",
        "perceptual_quality",
        "audio_video_sync",
        "safety",
    )

    # Editing task taxonomy (Fig. 1)
    task_types: tuple[str, ...] = (
        "identity_preserving_speech",
        "av_instance_edit",
        "av_instance_insert",
        "av_instance_remove",
    )

    ltx_hook: str = (
        "Instruction-only AV joint edit on LTX/Ovi latents: channel-concat source + SIGA + "
        "two-stage train; preserves ambient/non-target audio"
    )

    # Channel concat doubles latent channels per modality (Sec. 4.1)
    use_source_concat: bool = True
    use_siga: bool = True
    two_stage_training: bool = True
