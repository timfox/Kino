"""Framework card, NMED benchmarks, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.eeg_music.alignment import clip_contrastive_loss
from ltx_trainer.eeg_music.augmentation import augment_eeg_segment
from ltx_trainer.eeg_music.config import EegMusicConfig
from ltx_trainer.eeg_music.distillation import multi_view_distill_loss
from ltx_trainer.eeg_music.masking_theory import cross_cluster_overlap_ratio, kernel_ratio
from ltx_trainer.eeg_music.model import ChannelOrientedEEGStub


def framework_card(cfg: EegMusicConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EegMusicConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "github": cfg.github,
        "demo_page": cfg.demo_page,
        "datasets": ["NMED-T", "NMED-H"],
        "recording_hours": 29.4,
        "subjects": 68,
        "channels": cfg.num_channels,
        "align_dim": cfg.align_dim,
        "pipeline": ["channel-wise tokenization", "multi-view self-distillation", "channel dropout", "CLIP alignment", "ridge→CLAP→AudioLDM"],
        "music_encoder": "frozen CLAP (AudioLDM)",
        "decoder": "pretrained AudioLDM (no fine-tune)",
        "design_principle": "defer channel mixing; per-electrode tokens",
    }


def table1_main_results() -> list[dict[str, Any]]:
    """Table 1 — combined NMED-T + NMED-H."""
    return [
        {"method": "Linear EEG Reference", "CLAP": 0.576, "id_50way": 0.067, "id_14way": 0.067},
        {"method": "Audio Reconstruction Reference", "CLAP": 0.752, "id_50way": 0.598, "id_14way": 0.775},
        {"method": "EEG2Mel", "CLAP": 0.588, "id_50way": 0.259, "id_14way": 0.478},
        {"method": "LaBraM", "CLAP": 0.657, "id_50way": 0.380, "id_14way": 0.681},
        {"method": "EEGPT", "CLAP": 0.625, "id_50way": 0.326, "id_14way": 0.643},
        {"method": "CBraMod", "CLAP": 0.641, "id_50way": 0.402, "id_14way": 0.690},
        {"method": "Ours", "CLAP": 0.683, "id_50way": 0.487, "id_14way": 0.692},
    ]


def table2_ablation() -> list[dict[str, Any]]:
    """Table 2 — channel-oriented components."""
    return [
        {"variant": "Full model (Ours)", "id_50way": 0.487, "id_14way": 0.692},
        {"variant": "Block tokenization (g=5)", "id_50way": 0.141, "id_14way": 0.406},
        {"variant": "No multi-view pretraining", "id_50way": 0.050, "id_14way": 0.155},
        {"variant": "No channel dropout", "id_50way": 0.411, "id_14way": 0.598},
        {"variant": "Linear head, encoder fixed", "id_50way": 0.092, "id_14way": 0.191},
        {"variant": "Encoder fixed during alignment", "id_50way": 0.179, "id_14way": 0.425},
    ]


def theorem_42_smoke() -> dict[str, Any]:
    """Illustrate Lemma 4.1 ratio and covariance condition."""
    blocks = [{0, 1}, {2, 3}, {4}]
    diff = {0, 1, 2}
    ratio = kernel_ratio(diff, rho=0.5, num_coords=5, blocks=blocks)
    y_same = [True, False, False, True]
    s_d = [0, 2, 1, 0]
    r_ch, r_blk = cross_cluster_overlap_ratio(y_same, s_d, b=0.25)
    return {"kernel_ratio": ratio, "r_channel": r_ch, "r_block": r_blk, "channel_lower_overlap": r_ch < r_blk}


def forward_smoke(cfg: EegMusicConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EegMusicConfig()
    model = ChannelOrientedEEGStub(cfg)
    x = torch.randn(cfg.demo_batch, cfg.num_channels, cfg.demo_time_samples)
    z = model(x)
    x_aug = augment_eeg_segment(x, channel_drop_prob=cfg.channel_dropout)
    z_audio = torch.randn(cfg.demo_batch, cfg.align_dim)
    align_loss = clip_contrastive_loss(z, z_audio)
    t_logits = [torch.randn(cfg.demo_batch, 32) for _ in range(cfg.global_views)]
    s_logits = [torch.randn(cfg.demo_batch, 32) for _ in range(cfg.global_views + cfg.local_views)]
    distill = multi_view_distill_loss(t_logits, s_logits)
    return {
        "z_shape": list(z.shape),
        "aug_shape": list(x_aug.shape),
        "align_loss": float(align_loss.detach()),
        "distill_loss": float(distill.detach()),
        "theorem": theorem_42_smoke(),
    }


def evaluation_demo(cfg: EegMusicConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EegMusicConfig()
    return {
        "framework": framework_card(cfg),
        "table1": table1_main_results(),
        "table2": table2_ablation(),
        "headline": {
            "clap": cfg.clap_score,
            "id_50way": cfg.id_50way,
            "delta_50way_vs_cbramod": cfg.id_50way - cfg.baseline_cbramod_50way,
            "delta_50way_vs_eeg2mel": cfg.id_50way - cfg.eeg2mel_50way,
        },
        "forward": forward_smoke(cfg),
    }
