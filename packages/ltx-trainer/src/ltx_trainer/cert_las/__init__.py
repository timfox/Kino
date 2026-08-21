"""Cert-LAS: certified T2I diffusion model ownership verification (arXiv:2605.29809)."""

from ltx_trainer.cert_las.auditing import audit_backdoor_method, prompt_suspiciousness_sin
from ltx_trainer.cert_las.config import CertLASConfig
from ltx_trainer.cert_las.hooks import (
    attach_cert_las_to_preprocess_meta,
    caption_hint_lines,
    cert_las_enabled,
    cert_las_preprocess_extra,
    cert_las_user_prompt_lines,
    ltx_video_prompt_suffix,
    merge_preprocess_extra,
    qa_manifest_directory,
)
from ltx_trainer.cert_las.lfs import allocate_layer_sigmas, layer_fine_tuning_sensitivity
from ltx_trainer.cert_las.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
from ltx_trainer.cert_las.verify import ownership_threshold, verify_ownership

__all__ = [
    "CertLASConfig",
    "allocate_layer_sigmas",
    "attach_cert_las_to_preprocess_meta",
    "audit_backdoor_method",
    "caption_hint_lines",
    "cert_las_enabled",
    "cert_las_preprocess_extra",
    "cert_las_user_prompt_lines",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
    "layer_fine_tuning_sensitivity",
    "ltx_video_prompt_suffix",
    "merge_preprocess_extra",
    "ownership_threshold",
    "prompt_suspiciousness_sin",
    "qa_manifest_directory",
    "verify_ownership",
]
