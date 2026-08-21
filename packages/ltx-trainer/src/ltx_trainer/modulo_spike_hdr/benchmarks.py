"""Reference metrics from modulo spike HDR paper (arXiv:2604.14632)."""

from __future__ import annotations

from typing import Any

PAPER_ARXIV = "2604.14632"
PAPER_TITLE = (
    "High-Speed Full-Color HDR Imaging via Unwrapping Modulo-Encoded Spike Streams"
)

# Table I — UnModNet dataset synthetic test
TABLE1_SYNTHETIC: dict[str, dict[str, float]] = {
    "ours": {
        "psnr_l": 39.17,
        "ssim_l": 0.977,
        "hdr_vdp3": 8.837,
        "psnr_pu": 33.77,
        "ssim_pu": 0.974,
        "total_inference_s": 42.57,
        "per_frame_s": 0.27,
    },
    "unmodnet": {
        "psnr_l": 35.46,
        "ssim_l": 0.972,
        "hdr_vdp3": 8.687,
        "total_inference_s": 169.55,
    },
    "pnp_ua": {
        "psnr_l": 28.58,
        "ssim_l": 0.707,
        "total_inference_s": 53.42,
    },
}

# Table II — ablations (selected)
TABLE2_ABLATIONS: dict[str, dict[str, float]] = {
    "without_stage2": {"psnr_l": 34.33, "ssim_l": 0.920},
    "without_ccp_refiner": {"psnr_l": 38.64, "ssim_l": 0.975},
    "complete": {"psnr_l": 39.17, "ssim_l": 0.977},
}

HARDWARE_PROTOTYPE = {
    "readout_hz": 20_000,
    "output_fps": 1000,
    "stride_frames": 20,
    "window_frames": 25,
    "raw_spike_gbps": 20.0,
    "modulo_output_gbps": 6.0,
    "resolution": "500x500x3",
    "sensor": "Spike M1K40-H2-Gen3",
}

ARCHITECTURE = {
    "formulation": "exposure-decoupled representation + query (Eq. 3–4)",
    "stage1": "PMF-Adapter + frozen diffusion prior (LDM / SD1.5)",
    "stage2": "LMA-Decoder + CCP-Refiner with LAR physics (Eq. 7–18)",
    "hardware": "chromatic spike front-end + per-pixel modulo register back-end (Eq. 24)",
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_synthetic": TABLE1_SYNTHETIC,
        "table2_ablations": TABLE2_ABLATIONS,
        "hardware_prototype": HARDWARE_PROTOTYPE,
        "architecture": ARCHITECTURE,
    }
