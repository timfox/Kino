"""Neural UV Atlas configuration (Salehi arXiv:2606.10050)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.10050"
PAPER_TITLE = (
    "Continuous Neural Reparameterization as a Deep Geometric Prior "
    "for Robust Fixed-Chart UV Repair"
)
PAPER_AUTHORS = "Mohammad Sadegh Salehi (Zero One Creative)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_REPO = "https://github.com/01C-Amara/NeuralUVAtlas"
PAPER_VENUE = "Preprint (Jun 2026)"

# SIREN (Sec. 3, Sec. 4)
SIREN_LAYERS = 5
SIREN_WIDTH = 256
SIREN_OMEGA0 = 15
SPECTRAL_RANK = 16

# Objective (Eq. 5)
EPS_J = 1e-4
EPS_DET = 1e-3
BARRIER_ALPHA_START = 10.0
BARRIER_ALPHA_END = 0.1

# Optimization (Sec. 4)
TUTTE_WARMUP_ITERS = 200
ADAM_ITERS = 5000
LR_START = 5e-4
LR_END = 1e-6
NTK_SUBSAMPLE = 256

# Compact benchmark charts (Sec. 5)
COMPACT_CHARTS = ("Hand", "Bob", "Camel")


@dataclass
class NeuralUVConfig:
    siren_layers: int = SIREN_LAYERS
    siren_width: int = SIREN_WIDTH
    omega0: float = SIREN_OMEGA0
    spectral_rank: int = SPECTRAL_RANK
    tutte_warmup_iters: int = TUTTE_WARMUP_ITERS
    adam_iters: int = ADAM_ITERS
    eps_j: float = EPS_J
    eps_det: float = EPS_DET
    barrier_alpha_start: float = BARRIER_ALPHA_START
    barrier_alpha_end: float = BARRIER_ALPHA_END
    ntk_subsample: int = NTK_SUBSAMPLE
