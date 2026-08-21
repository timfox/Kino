"""ST-SFLora pipeline stages and stub limitations."""

from __future__ import annotations

PIPELINE_STAGES: tuple[str, ...] = (
    "mobility_aware_client_selection",
    "downlink_model_broadcast",
    "client_forward_tokenization",
    "upload_batch_token_importance",
    "server_joint_opt_K_W_p",
    "selected_token_uplink",
    "server_lora_finetune",
)

LIMITATIONS: tuple[str, ...] = (
    "No real ViT/LoRA training or timm weights — numpy toy attention and STE only.",
    "No Rayleigh fading trace or Poisson client sampling — static toy CSI.",
    "Alternating optimization is reference; not certified globally optimal for P0.",
    "Tables I–II are paper excerpts, not reproduced FL runs.",
)
