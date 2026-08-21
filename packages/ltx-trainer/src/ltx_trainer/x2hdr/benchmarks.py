"""Reference metrics — X2HDR (arXiv:2602.04814) + LumiVid Table 2 cross-refs."""

from __future__ import annotations

from typing import Any

PAPER_ARXIV = "2602.04814"
PAPER_TITLE = "X2HDR: HDR Video Generation from SDR Video with Diffusion Models"

# Table 1 — PU21 VAE roundtrip (synthetic anchor from ingest smoke)
TABLE1_PU21_VAE: dict[str, float] = {
    "pu21_psnr": 33.5,
    "pu21_ssim": 0.962,
    "kl_lat": 0.012,
}

# Table 2 — ARRI / UPIQ (from LumiVid Table 2 x2hdr row)
TABLE2_ARRI: dict[str, float] = {"pu21_psnr": 20.68, "lpips": 0.250, "jod": 3.54}
TABLE2_UPIQ: dict[str, float] = {"pu21_psnr": 17.47, "lpips": 0.177, "jod": 6.06}

# Table 3 — temporal (LumiVid Table 3 x2hdr row)
TABLE3_TEMPORAL: dict[str, float] = {"f2f_psnr": 36.36, "flicker": 0.1630, "jod": 3.54}

TRAINING_DEFAULTS = {
    "steps": 10_000,
    "lora_param_fraction": 0.01,
    "denoise_steps_inference": 11,
    "vae_encoding": "pu21",
    "l_peak_cd_m2": 4000.0,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_pu21_vae": TABLE1_PU21_VAE,
        "table2_arri": TABLE2_ARRI,
        "table2_upiq": TABLE2_UPIQ,
        "table3_temporal": TABLE3_TEMPORAL,
        "training_defaults": TRAINING_DEFAULTS,
    }
