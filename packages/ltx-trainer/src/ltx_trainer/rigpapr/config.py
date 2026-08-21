"""RigPAPR configuration (Peng et al. arXiv:2606.06685)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.06685"
PAPER_TITLE = (
    "RigPAPR: Rig-Based Animation of Static Neural Point Clouds "
    "from a Fixed-Viewpoint Video"
)
PAPER_AUTHORS = "Shichong Peng, Yanshu Zhang, Ke Li (SFU APEX Lab)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "Preprint (Jun 2026)"

# Auto-rig pipeline (Sec. 3.3, App. C.2)
AUTO_RIG_MESH = "OffsetOPT"
AUTO_RIG_SKELETON = "Puppeteer"
IDW_NEIGHBOURS = 6

# PAPR renderer (Sec. 3.2)
PAPR_TOP_K = 8
FEATURE_DIM = 32

# Optimization (App. C.3)
PHASE1_ITERS_PER_FRAME = 2000
PHASE2_ITERS = 30_000
PHASE1_LR = (5e-3, 1e-4)
PHASE2_LR_MLP = (3e-4, 1e-5)
PHASE2_LR_WEIGHTS = (1e-2, 1e-5)

# Synthetic / real resolution (Sec. 4.1)
SYNTH_RES = (512, 512)
REAL_RES = (960, 540)

# Scene frame counts (App. Table 4)
SCENE_FRAMES: dict[str, int] = {
    "t_rex": 40,
    "simpsons": 34,
    "fox": 24,
    "wolf": 18,
    "spider": 16,
    "robot": 36,
    "statue": 18,
}


@dataclass
class RigPAPRConfig:
    num_points: int = 4096
    num_bones: int = 24
    top_k: int = PAPR_TOP_K
    feature_dim: int = FEATURE_DIM
    idw_k: int = IDW_NEIGHBOURS
    arap_k: int = 8
    phase1_iters: int = PHASE1_ITERS_PER_FRAME
    phase2_iters: int = PHASE2_ITERS
    depth_subsample: int = 4096
    depth_threshold: float = 0.05
