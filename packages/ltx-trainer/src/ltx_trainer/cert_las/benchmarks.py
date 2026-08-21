"""Paper table excerpts for Cert-LAS (arXiv:2605.29809)."""

from __future__ import annotations

from typing import Any

# Table 2 — main comparison (σ_k = 1.0 row excerpt)
TABLE2_FIDELITY = {
    "None": {"fid": 25.23, "clip": 31.22, "dreamsim": None},
    "WatermarkDM": {"fid": 26.27, "clip": 30.86, "dreamsim": 0.164},
    "SleeperMark": {"fid": 25.85, "clip": 31.21, "dreamsim": 0.120},
    "Cert-LAS (w/o smooth)": {"fid": 25.91, "clip": 31.13, "dreamsim": 0.111},
    "Vanilla (σ1.0)": {"fid": 27.48, "clip": 31.05, "dreamsim": 0.123},
    "Cert-LAS (w/ smooth)": {"fid": 25.91, "clip": 31.13, "dreamsim": 0.111},
}

TABLE2_WATERMARK = {
    "WatermarkDM": {"t_at_1e6": 0.879, "sin_p": 0.964, "sout_p": 0.933},
    "SleeperMark": {"t_at_1e6": 0.998, "sin_p": 0.958, "sout_p": 0.025},
    "Cert-LAS (w/o smooth)": {"t_at_1e6": 1.000, "sin_p": 0.125, "sout_p": 0.016},
    "Cert-LAS (w/ smooth)": {"t_at_1e6": 1.000, "sin_p": 0.125, "sout_p": 0.016, "vsr": 1.000, "R_bar": 1.48},
}

# Table 1 — auditing (paper §3.1)
TABLE1_AUDITING = {
    "WatermarkDM": {"sin_p": 0.9642, "sout_p": 0.9330},
    "SleeperMark": {"sin_p": 0.9581, "sout_p": 0.0245},
}

# Table 7 — ℓ2 perturbation at budget 0.8
TABLE7_L2 = {
    "WatermarkDM": 0.000,
    "SleeperMark": 0.000,
    "Cert-LAS (w/o smooth)": 0.000,
    "Cert-LAS (w/ smooth)": 0.965,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_auditing": TABLE1_AUDITING,
        "table2_fidelity": TABLE2_FIDELITY,
        "table2_watermark": TABLE2_WATERMARK,
        "table7_l2_08": TABLE7_L2,
        "cert_las_beats_backdoor_stealth": TABLE2_WATERMARK["Cert-LAS (w/o smooth)"]["sin_p"]
        < TABLE1_AUDITING["WatermarkDM"]["sin_p"],
        "cert_las_robust_l2": TABLE7_L2["Cert-LAS (w/ smooth)"] >= 0.965,
        "certified_radius_sigma1": TABLE2_WATERMARK["Cert-LAS (w/ smooth)"]["R_bar"],
    }
