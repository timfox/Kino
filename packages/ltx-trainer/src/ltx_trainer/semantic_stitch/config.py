"""SemanticStitch configuration (Jin et al. arXiv:2511.12084)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2511.12084"
PAPER_TITLE = "SemanticStitch: Enhancing Image Coherence through Foreground-Aware Seam Carving"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/Pokerman8/OAIV-Coherence"

# Paper training / inference (Sec. 4.1)
TRAIN_EPOCHS = 100
TRAIN_GPU_GB = 7.3
INFER_SIZE = 512
INFER_SEC = 0.114
PARAMS_M = 33.55
GFLOPS = 10.77


@dataclass
class SemanticStitchConfig:
    image_size: int = INFER_SIZE
    train_epochs: int = TRAIN_EPOCHS
    max_mask_epochs: int = 50
    smooth_weight: float = 1.0
    exclusivity_weight: float = 1.0
