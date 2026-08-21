"""LTX training plan for Mirage latent spatial memory sidecar."""
from __future__ import annotations

from typing import Any

from ltx_trainer.mirage.config import MirageConfig


def ltx_training_plan(cfg: MirageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MirageConfig()
    return {
        "fold": "mirage",
        "paper": cfg.paper_arxiv,
        "backbone": cfg.backbone,
        "objective": "flow_matching_with_latent_memory_control",
        "phases": [
            {
                "id": "stage1_control_branch",
                "train": ["control_side_branch"],
                "freeze": ["backbone", "vae"],
                "lr": cfg.stage1_lr,
            },
            {
                "id": "stage2_lora_joint",
                "train": ["control_side_branch", "lora_self_attn"],
                "freeze": ["vae"],
                "lr": cfg.stage2_lr,
                "lora_rank": cfg.lora_rank,
            },
        ],
        "chunk_latent_frames": cfg.chunk_latent_frames,
        "control_layers": list(cfg.control_layers),
        "trainer_flags": {
            "use_mirage_weights": True,
            "min_mirage_readout_weight": 0.35,
            "note": "Upweight shards where fold annotator reports high mirage.readout_coverage",
        },
        "fold_hooks": ["ltx_trainer.mirage.fold.annotate_video_latent_data"],
        "fold_hook": "ltx_trainer.mirage.fold.annotate_video_latent_data",
        "notes": "Latent cache readout injected at VACE-style side branch; no rasterize-and-encode loop.",
    }


def gopex_env_exports() -> dict[str, str]:
    """Shell-friendly env for Mirage fold backfill + world-model training."""
    return {
        "GOPEX_AV_FOLD_HOOKS": "mirage,phyworld,avbench,longav_compass",
        "GOPEX_ENABLE_AV_FOLD": "1",
        "GOPEX_AV_FOLD_TRAIN": "1",
    }
