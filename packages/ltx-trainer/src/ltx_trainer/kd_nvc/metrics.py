"""Paper table anchors — Tables III–VI, Fig. 1."""

from __future__ import annotations

from typing import Any


def operating_guidelines() -> list[str]:
    return [
        "Decoder-side KD-NVC: AE-NAS selects per-module student, then one-step EFD distillation.",
        "Exclude entropy model from architecture-level search (high sensitivity).",
        "Rank architecture combos by η̂ = S / ΣΔ without full codec training.",
        "EFD aligns adaptive-pooled channel-energy signatures (k=8), not pixel MSE.",
        "Set β=1.0 in stage 1, β=0.0 afterward for faster RD convergence.",
        "Compare BD-rate vs VTM-LDB-23.11 on UVG + HEVC B–E + MCL-JCV.",
    ]


def table_bd_rate_ip32() -> dict[str, Any]:
    """Table III averages @ ≈60% and ≈100% speed-up (YUV420)."""
    return {
        "anchor": "VTM-LDB-23.11",
        "teacher_dcvc_rt_repro": -15.3,
        "speedup_60pct": {
            "direct_training_avg": -5.4,
            "fu2024_avg": -6.8,
            "prim_avg": -7.2,
            "smodi_avg": -7.4,
            "kd_nvc_s_avg": -9.0,
        },
        "speedup_100pct": {
            "direct_training_avg": 3.0,
            "fu2024_avg": 1.2,
            "prim_avg": 0.7,
            "smodi_avg": 0.8,
            "kd_nvc_t_avg": -1.5,
        },
    }


def table_bd_rate_ip_minus1() -> dict[str, Any]:
    return {
        "speedup_60pct": {
            "smodi_avg": -9.4,
            "kd_nvc_s_avg": -11.1,
        },
        "speedup_100pct": {
            "smodi_avg": 2.2,
            "kd_nvc_t_avg": 0.7,
        },
    }


def table_complexity() -> dict[str, Any]:
    """Table IV @ 1080p."""
    return {
        "dcvc_rt": {"params_m": 20.7, "dec_kmacs": 166.8, "enc_kmacs": 142.4},
        "kd_nvc_s": {"params_m": 16.0, "dec_kmacs": 93.7, "dec_reduction_pct": 44},
        "kd_nvc_t": {"params_m": 14.6, "dec_kmacs": 71.7, "dec_reduction_pct": 57},
    }


def table_fps_rtx5060_1080p() -> dict[str, Any]:
    """Table V decode FPS highlights."""
    return {
        "dcvc_rt_decode_fps": 31.2,
        "kd_nvc_s_decode_fps": 53.2,
        "kd_nvc_t_decode_fps": 69.2,
        "realtime_threshold_fps": 30,
        "high_framerate_fps": 60,
    }


def table_distill_cost() -> dict[str, Any]:
    """Table VI iteration time / memory on 384×256 patch."""
    return {
        "direct_training_ms": 69.3,
        "mse_ms": 86.7,
        "efd_ms": 85.4,
        "frequency_domain_ms": 482.1,
    }


def fig1_gain_anchors() -> dict[str, Any]:
    return {
        "ae_nas_vs_uniform_layer_bdrate_gap_pct": 3.5,
        "efd_vs_smodi_bdrate_gap_pct": 1.6,
        "kd_nvc_t_vs_vtm_ldb_ip32": -1.5,
    }
