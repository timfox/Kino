"""Multi-GPU 3D FDTD+CPML communication study (Obieke, arXiv:2606.06910)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06910"
PAPER_TITLE = (
    "Communication Strategy Selection for Multi-GPU 3D FDTD "
    "with Convolutional Perfectly Matched Boundary Layers"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
UPSTREAM_REPO = "https://github.com/victoryobieke/fdtd-cpml-multigpu"

STENCIL_RADIUS = 4  # eighth-order
LAMBDA_FDTD = 0.05
L_CPML = 20
R_TARGET = 1e-8
KAPPA_MAX = 5.0
ALPHA_FRAC = 0.05
COMM_INTERVALS = (1, 2, 4, 8)

DECOMPOSITIONS = (
    ("slab_z", "1×1×4"),
    ("block_xy", "2×2×1"),
    ("pencil_yz", "1×2×2"),
)

PEER_SPEEDUP_RANGE = (2.46, 2.76)
SINGLE_GPU_MPOINTS_RANGE = (2889.58, 3304.48)
CPML_OVERHEAD_MAX_PCT = 1.0
BEST_ENLARGED_GHOST_S = 4
