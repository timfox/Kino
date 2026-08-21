"""Paper anchors — dilated symmetric difference (Urieli, arXiv:2606.06512)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06512"
PAPER_TITLE = "Dilated Symmetric Difference for Binary Image Comparison"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_AUTHOR = "Sharon Urieli"

# Fig. 1 — die-3 vs misaligned die-4 (D3 dilate + shift (4,2))
FIG1_DELTA_ALIGN = 3 + (4**2 + 2**2) ** 0.5  # ≈ 7.5
FIG1_OPTIMAL_R = 8
FIG1_IOU_AT_R8 = 1.0

# Fig. 3 — IoU vs r curve anchors (die disks far apart)
FIG3_IOU_CURVE: tuple[tuple[int, float], ...] = (
    (0, 0.0),  # r=0: symmetric diff shows alignment error
    (7, 0.95),
    (8, 1.0),
    (30, 0.4),  # D30 extends into adjacent disks
)

# Fig. 4 — rotation θ=1° + elastic warp (σ=8, α=30)
FIG4_WARP = {"theta_deg": 1.0, "sigma": 8.0, "alpha": 30.0}
FIG4_IOU_CURVE: tuple[tuple[int, float], ...] = (
    (0, 0.0),
    (3, 0.85),
    (4, 0.92),  # misalignment gone but gaps l≤8 missed
    (6, 0.75),  # central disk engulfed
)

RADIUS_RULES = {
    "lower_bound": "r > δ_align",
    "upper_bound": "d(boundary, detected region) > r",
    "enclosed_regions": "largest dimension > 2r",
}

TRAIN_DEFAULTS = {
    "operator": "A ⊕^r_△ B = (A ∩ (B ⊕ D_r)̄) ∪ (B ∩ (A ⊕ D_r)̄)",
    "symmetric_diff": "A△B = (A ∩ B̄) ∪ (B ∩ Ā)",
    "extension": "differentiable morphological layer with learnable r",
}
