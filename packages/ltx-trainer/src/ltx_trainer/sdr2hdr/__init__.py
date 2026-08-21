"""SDR → HDR video via exposure bracketing + merge (Tedla et al. arXiv:2605.14703)."""

from ltx_trainer.sdr2hdr.pipeline import (
    Sdr2HdrConfig,
    sdr_gamma_soft_shoulder,
    sdr_video_to_hdr,
    tone_map_hdr_for_preview,
)
from ltx_trainer.sdr2hdr.vmm import VideoMergingModel, merge_brackets_debevec, vmm_log_loss

__all__ = [
    "Sdr2HdrConfig",
    "VideoMergingModel",
    "merge_brackets_debevec",
    "sdr_gamma_soft_shoulder",
    "sdr_video_to_hdr",
    "tone_map_hdr_for_preview",
    "vmm_log_loss",
]
