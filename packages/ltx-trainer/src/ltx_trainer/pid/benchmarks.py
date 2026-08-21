"""Paper table anchors — PiD (arXiv:2605.23902)."""

from __future__ import annotations

from typing import Any

# Table 1 — PiD rows (GB200 compile latency ms, selected metrics)
TABLE1: dict[str, dict[str, float]] = {
    "flux1_vae_flux1_dev_pid_24_28": {
        "musiq": 73.26,
        "niqe": 3.50,
        "deqa": 4.31,
        "uni_iaa": 66.21,
        "uni_iqa": 75.21,
        "vq_r1": 4.68,
        "latency_compile_ms": 211.2,
    },
    "sd3_vae_sd3_medium_pid_24_28": {
        "musiq": 74.00,
        "niqe": 3.11,
        "deqa": 4.26,
        "uni_iaa": 62.57,
        "uni_iqa": 74.22,
        "vq_r1": 4.59,
        "latency_compile_ms": 214.0,
    },
    "flux2_vae_flux2_dev_pid_45_50": {
        "musiq": 73.79,
        "niqe": 3.12,
        "deqa": 4.30,
        "uni_iaa": 66.01,
        "uni_iqa": 75.71,
        "vq_r1": 4.66,
        "latency_compile_ms": 206.1,
    },
    "flux1_vae_z_image_pid_45_50": {
        "musiq": 74.08,
        "niqe": 3.26,
        "deqa": 4.29,
        "uni_iaa": 63.96,
        "uni_iqa": 75.23,
        "vq_r1": 4.64,
        "latency_compile_ms": 211.2,
    },
    "dinov2_ditdh_pid_50_50": {
        "musiq": 73.31,
        "niqe": 3.38,
        "deqa": 4.27,
        "uni_iaa": 69.81,
        "uni_iqa": 76.52,
        "vq_r1": 4.63,
        "latency_compile_ms": 212.4,
    },
    "siglip_scale_rae_pid_50_50": {
        "musiq": 74.03,
        "niqe": 3.34,
        "deqa": 4.17,
        "uni_iaa": 64.94,
        "uni_iqa": 72.78,
        "vq_r1": 4.45,
        "latency_compile_ms": 208.7,
    },
}

# Best cascaded baseline (VAE + InvSR compile) for FLUX.1 dev — comparison
TABLE1_BASELINE_FLUX1: dict[str, float] = {
    "musiq": 74.11,
    "niqe": 3.82,
    "latency_compile_ms": 1017.7,
    "method": "vae_dec_invsr",
}

# Table 2 — teacher vs student (FLUX.1 dev, PiD 24/28)
TABLE2: dict[str, dict[str, float]] = {
    "teacher_50": {"musiq": 71.79, "niqe": 4.92, "psnr": 24.96, "lpips": 0.16},
    "teacher_25": {"musiq": 71.63, "niqe": 5.43, "psnr": 25.00, "lpips": 0.18},
    "teacher_4": {"musiq": 68.32, "niqe": 7.00, "psnr": 25.70, "lpips": 0.21},
    "student_4": {"musiq": 73.26, "niqe": 3.50, "psnr": 24.19, "lpips": 0.09},
}

# Table 3 — latency ms (compile) and memory GB @ 2048
TABLE3_LATENCY_MS: dict[str, dict[str, float]] = {
    "rtx_5090": {"512": 78.4, "1024": 188.2, "2048": 979.3, "4096": 9238.0},
    "h100": {"512": 45.3, "1024": 88.4, "2048": 446.0, "4096": 3754.6},
    "gb200": {"512": 33.0, "1024": 57.0, "2048": 208.8, "4096": 1927.3},
}

TABLE3_MEMORY_GB: dict[str, dict[str, float]] = {
    "pid_compile": {"2048": 13.0, "4096": 22.5},
    "flux1_vae_compile": {"2048": 16.7},
}

# Table 4 — ablations (FLUX.1 dev)
TABLE4: dict[str, dict[str, float]] = {
    "w/o_t2i_prior": {"musiq": 59.52, "niqe": 7.79, "vq_r1": 2.587},
    "w/o_sigma_gate": {"musiq": 70.84, "niqe": 5.84, "vq_r1": 4.647},
    "ours": {"musiq": 71.63, "niqe": 5.43, "vq_r1": 4.649},
}

# Fig. 4 — MLLM win rates vs cascaded SR (approximate)
FIG4_WIN_RATES: dict[str, dict[str, float]] = {
    "claude_opus_4_6": {"pid": 85.4, "consistency": 84.3},
    "gemini_3_flash": {"pid": 90.1, "consistency": 88.0},
    "gpt_5_5": {"pid": 80.5, "consistency": 79.9},
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "table1_baseline_flux1": TABLE1_BASELINE_FLUX1,
        "table2": TABLE2,
        "table3_latency_ms": TABLE3_LATENCY_MS,
        "table3_memory_gb": TABLE3_MEMORY_GB,
        "table4": TABLE4,
        "fig4_win_rates": FIG4_WIN_RATES,
        "claims": {
            "512_to_2048_under_1s_rtx5090": True,
            "peak_memory_gb_2048": 13.0,
            "gb200_compile_ms_2048": 211.2,
            "scale_factors": [4, 8],
        },
        "website": "https://research.nvidia.com/labs/sil/projects/pid/",
    }
