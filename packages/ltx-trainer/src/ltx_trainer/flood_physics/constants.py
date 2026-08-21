"""Paper anchors — physics-guided UNet+FNO flood prediction (Gebre et al., arXiv:2606.06524)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06524"
PAPER_TITLE = (
    "Advanced Flood Prediction with Physics-Guided Deep Learning: "
    "Combining UNet, FNO, and SAR/Optical Imagery"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "IEEE Radar Conference (RadarConf 2026)"

INPUT_MODALITIES = ("sentinel1_sar", "sentinel2_optical", "dem_terrain")

# Table II — flood extent prediction
TABLE2_EXTENT: tuple[dict[str, str | float], ...] = (
    {"model": "UNet-only", "IoU": 0.75, "F1": 0.85},
    {"model": "FNO-only", "IoU": 0.77, "F1": 0.87},
    {"model": "Hybrid (UNet + FNO)", "IoU": 0.82, "F1": 0.90, "IoU_ci": "±0.03", "F1_ci": "±0.02"},
)

HYBRID_ROW = next(r for r in TABLE2_EXTENT if "Hybrid" in str(r["model"]))

# Continuous field RMSE (HEC-RAS reference)
HYDRO_METRICS = {
    "depth_rmse_m": 0.21,
    "velocity_rmse_m_s": 0.15,
    "unet_depth_rmse_m": 0.34,
    "fno_depth_rmse_m": 0.28,
    "mass_residual_mse": 0.03,
    "momentum_residual_mse": 0.05,
    "mass_imbalance_pct": 2.1,
}

# Physics ablation
PHYSICS_ABLATION_DEPTH_RMSE_INCREASE_PCT = 18.0

TRAIN_DEFAULTS = {
    "optimizer": "AdamW",
    "pretrain": "UNet extent first, then joint UNet+FNO",
    "lambda_phys_ramp": "0 → target over initial epochs",
    "mixed_precision": True,
    "inputs": list(INPUT_MODALITIES),
}
