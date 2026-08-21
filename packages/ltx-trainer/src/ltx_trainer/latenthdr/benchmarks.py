"""Reference metrics from LatentHDR paper tables (arXiv:2605.11115)."""

from __future__ import annotations

from typing import Any

# Table 1 — no-reference synthetic (Ours variants, v2 = without blend post-processing)
TABLE1_NO_REF: dict[str, dict[str, Any]] = {
    "ours_l2h_v2": {
        "perspective_stops": (9.93, 2.47),
        "panoramic_stops": (10.29, 2.77),
        "perspective_pu21": (46.20, 15.16),
        "diffusion_runs": 0,
        "latency_s": 0.28,
        "mem_gb": 2,
    },
    "ours_t2h_v2": {
        "perspective_stops": (10.38, 2.52),
        "panoramic_stops": (10.64, 2.77),
        "perspective_pu21": (46.67, 15.26),
        "diffusion_runs": 1,
        "latency_s": 2.35,
        "mem_gb": 35.6,
    },
    "lediff_v2": {
        "perspective_stops": (4.32, 1.06),
        "panoramic_stops": (4.70, 0.93),
        "diffusion_runs": 2,
        "latency_s": 2.62,
    },
    "bd_glide": {
        "perspective_stops": (7.02, 1.86),
        "diffusion_runs": 5,
        "latency_s": 120.0,
    },
}

# Table 2 — SI-HDR reference-based (Ours-v2)
TABLE2_SI_HDR: dict[str, tuple[float, float] | float] = {
    "ours_v2_stops": (10.3, 2.4),
    "ours_v2_pu21": (35.9, 12.3),
    "ours_v1_stops": (11.3, 3.7),
    "lediff_v2_stops": (4.5, 0.5),
    "hdrcnn_stops": (9.0, 2.7),
}

# Table 3 — ablation (reference row)
TABLE3_ABLATION: dict[str, dict[str, float]] = {
    "reference": {"stops": 11.37, "pu21": 40.43, "plow": 0.00, "phigh": 3.80},
    "step_0.5": {"stops": 11.37, "pu21": 40.58},
    "step_2": {"stops": 11.17, "pu21": 39.99},
    "range_-3_3": {"stops": 10.86, "pu21": 39.16},
    "w/o_ev_film": {"stops": 10.99, "pu21": 39.32},
}

# VAE posterior concentration (Sec. 3.2, FLUX.1-dev)
VAE_POSTERIOR_STATS: dict[str, float] = {
    "sigma_mean": 1.1e-4,
    "sigma_max": 9.7e-3,
    "sample_rmse_vs_mean": 2.9e-4,
}

PAPER_ARXIV = "2605.11115"
PAPER_TITLE = (
    "LatentHDR: Decoupling Exposure from Diffusion via Conditional "
    "Latent-to-Latent Mapping for Text/Image-to-Panoramic HDR"
)


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_no_ref": TABLE1_NO_REF,
        "table2_si_hdr": TABLE2_SI_HDR,
        "table3_ablation": TABLE3_ABLATION,
        "vae_posterior": VAE_POSTERIOR_STATS,
        "training": {
            "poly_haven_scenes": 954,
            "ev_range": (-7, 5),
            "default_ev_step": 1.0,
            "inference_diffusion_steps_t2h": 28,
        },
    }
