"""BTS-CAFE: FedDG for stethoscope-induced RSC shortcuts (arXiv:2605.29862)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BTSCafeConfig:
    paper_arxiv: str = "arXiv:2605.29862"
    backbone: str = "BTS"
    sample_rate_hz: int = 16000
    # FedDG
    num_clients: int = 3
    fed_rounds: int = 30
    local_epochs: int = 1
    learning_rate: float = 5e-5
    # GIN (Sec. 3.2.1)
    t_aug: int = 5
    t_w: int = 5
    gain_min: float = 0.8
    gain_max: float = 1.2
    alpha_min: float = 0.25
    # Text augmentation (Sec. 3.2.2)
    p_text_neutralize: float = 0.25
    # Gradient alignment (Sec. 3.2.3)
    lambda_align: float = 1e-3
    # Fig. 2 CLAP embedding analysis anchors
    raw_device_acc: float = 0.9354
    raw_disease_acc: float = 0.7294
    mean_device_acc: float = 0.8786
    mean_disease_acc: float = 0.7301
    whiten_device_acc: float = 0.6953
    whiten_disease_acc: float = 0.7202
    # Table 2 Setting #1 OOD ICBHI Score anchors (AKGC417L, Meditron, Yunting)
    ood_score_akgc417l: float = 52.82
    ood_score_meditron: float = 54.60
    ood_score_yunting: float = 65.69
    ood_score_littc2se: float = 43.15
    ood_score_litt3200: float = 66.24
