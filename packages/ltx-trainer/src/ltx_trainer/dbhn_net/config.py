"""DBHN-Net dual-branch hybrid SE — Fan et al., arXiv:2606.05911."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DbhnNetConfig:
    paper_arxiv: str = "arXiv:2606.05911"
    title: str = (
        "DBHN-Net: Dual-Branch Hybrid Neural Network For "
        "Low-Complexity Monaural Speech Enhancement"
    )
    framework: str = "DBHN-Net"

    # Training (§IV-B)
    sample_rate_hz: int = 16000
    fft_points: int = 320
    freq_bins: int = 161
    tf_mamba_blocks: int = 4
    tf_mamba_hidden: int = 128
    loss_beta: float = 0.5  # RI vs Mag (Eq. 45)
    lif_alpha: float = 10.0  # gradient proxy steepness (Eq. 41)

    # Table II — dual-branch ablation on WSJ0 (PESQ / ESTOI% / SI-SDR dB)
    full_pesq: float = 3.17
    full_estoi: float = 83.46
    full_sisdr: float = 12.16
    wo_ann_pesq: float = 2.75
    wo_snn_pesq: float = 2.81

    # Table III — Mamba vs LSTM/Transformer (MACs G/s)
    macs_dbhn: float = 1.32
    macs_lstm: float = 8.68
    macs_transformer: float = 22.05

    # Table IX — complexity anchors (MACs G/s on 1 s clip)
    macs_bsdb: float = 1.68
    macs_gag: float = 2.81
    complexity_reduction_x: float = 7.5

    # Table VI — WSJ0+DNS (AVG)
    wsj0_pesq_avg: float = 3.14
    wsj0_estoi_avg: float = 81.32
    wsj0_sisdr_avg: float = 11.93

    # Table VII — VoiceBank+Demand
    vb_wb_pesq: float = 3.08
    vb_stoi: float = 95.0
    vb_csig: float = 4.32
    vb_cbak: float = 3.59
    vb_covl: float = 3.74

    # Table VIII — DNS-Challenge
    dns_wb_pesq: float = 3.45
    dns_nb_pesq: float = 3.74
    dns_stoi: float = 98.0
    dns_sisdr: float = 20.63
