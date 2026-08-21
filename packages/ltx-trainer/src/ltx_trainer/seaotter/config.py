"""SEAOTTER cloud-robotics compression (Jacobellis & Yadwadkar, arXiv:2606.03940)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SeaotterConfig:
    paper_arxiv: str = "arXiv:2606.03940"
    title: str = (
        "SEAOTTER: Sensor Embedded Autoencoding with "
        "One-Time Transcode for Efficient Reconstruction"
    )
    repo_url: str = "https://github.com/UT-SysML/seaotter"
    frappe_repo: str = "https://github.com/UT-SysML/FRAPPE"

    # FRAPPE sensor encoder (frozen)
    frappe_rate_channels: tuple[int, ...] = (3, 6, 9, 12, 15)
    frappe_encoder_mac_per_pixel: tuple[float, float] = (10.0, 100.0)
    frappe_decoder_params_m: float = 57.0

    # Learned JPEG sandwich (K=3 headline rates)
    num_rate_points: int = 3
    dct_block_size: int = 8
    jpeg_subsampling: int = 0  # 4:4:4 — nonstandard learned color space
    lagrange_multipliers: tuple[float, float, float] = (0.75, 0.40, 0.22)
    rate_loss_weights: tuple[float, float, float] = (0.3, 0.7, 1.5)

    # Training recipe (Appendix A.8)
    train_dataset: str = "LSDIR"
    train_crop_size: int = 480
    train_epochs: int = 4
    train_lr: float = 1.2e-2
    train_batch_per_gpu: int = 4
    train_gpus: int = 4

    # Deployment tier thresholds (§3, Table 8)
    tier_ble_cr_min: float = 288.0
    tier_ble_encode_mpx_s: float = 12.0
    tier_5g_cr_min: float = 133.0
    tier_5g_encode_mpx_s: float = 28.0
    tier_wifi_cr_min: float = 60.0
    tier_wifi_encode_mpx_s: float = 62.0

    # Matched-rate operating point (Table 1, n=12 / FRAPPE)
    matched_frappe_n: int = 12
    matched_transmit_bpp_cls: float = 0.1086
    compression_ratio_at_200: float = 200.0  # paper headline @ CR 200:1

    task_names: tuple[str, ...] = ("cls", "seg", "clip")
