"""ID-LoRA configuration defaults aligned with upstream CelebV-HQ / TalkVid recipes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IdLoraConfig:
    paper_arxiv: str = "2603.10256"
    paper_title: str = "ID-LoRA: Identity-Driven Audio-Video Personalization with In-Context LoRA"
    project_url: str = "https://id-lora.github.io"
    submodule_path: str = "ID-LoRA"

    lora_rank: int = 128
    ref_audio_seconds: float = 6.0
    use_negative_ref_positions: bool = True
    mask_cross_attention_to_reference: bool = True
    mask_reference_from_text_attention: bool = True
    first_frame_conditioning_p: float = 0.9

    default_video_guidance: float = 3.0
    default_audio_guidance: float = 7.0
    default_identity_guidance: float = 3.0
    default_inference_steps: int = 30
    default_resolution: tuple[int, int] = (512, 512)
    default_num_frames: int = 121
    default_frame_rate: float = 25.0

    hf_checkpoints: dict[str, str] = field(
        default_factory=lambda: {
            "celebvhq_ltx2": "AviadDahan/ID-LoRA-CelebVHQ",
            "talkvid_ltx2": "AviadDahan/ID-LoRA-TalkVid",
            "celebvhq_ltx23": "AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K",
            "talkvid_ltx23": "AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K",
        }
    )
    hf_datasets: dict[str, str] = field(
        default_factory=lambda: {
            "celebvhq": "noakraicer/ID-LoRA-CelebVHQ",
            "talkvid": "noakraicer/ID-LoRA-TalkVid",
        }
    )
