"""Reference results (Yang et al., arXiv:2512.17343)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mddn.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — ODI-SR test, ×4 (PSNR / WS-PSNR)
TABLE1_ODI_SR_X4: dict[str, dict[str, float]] = {
    "GDGT-OSR": {"PSNR": 27.29, "WS-PSNR": 26.59},
    "OSRT": {"PSNR": 27.28, "WS-PSNR": 26.58},
    "MDDN": {"PSNR": 27.39, "WS-PSNR": 26.68},
}

# Table 1 — Flickr360 val, ×4
TABLE1_FLICKR_X4: dict[str, dict[str, float]] = {
    "GDGT-OSR": {"PSNR": 30.24, "WS-PSNR": 29.71},
    "MDDN": {"PSNR": 30.44, "WS-PSNR": 29.87},
}

# Table 2 — Flickr360 val, ×4 ablation (variants)
TABLE2_VARIANTS_X4: dict[str, dict[str, float]] = {
    "{1}": {"PSNR": 30.25, "WS-PSNR": 29.72, "Params_M": 11.73},
    "{1,2,3}": {"PSNR": 30.44, "WS-PSNR": 29.87, "Params_M": 13.04},
}

# Table 5 — fusion
TABLE5_FUSION_FLICKR_X4: dict[str, float] = {
    "Addition": 30.38,
    "MFF": 30.44,
}

# Table 6 — Flickr360 val ×8 complexity
TABLE6_COMPLEXITY: dict[str, dict[str, float]] = {
    "GDGT-OSR": {"Params_M": 13.45, "MultAdds_G": 56.69, "PSNR": 26.94, "WS-PSNR": 26.36},
    "MDDN": {"Params_M": 13.19, "MultAdds_G": 55.43, "PSNR": 27.05, "WS-PSNR": 26.45},
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1_odi_sr_x4": TABLE1_ODI_SR_X4,
        "table1_flickr_x4": TABLE1_FLICKR_X4,
        "table2_variants_x4": TABLE2_VARIANTS_X4,
        "table5_fusion": TABLE5_FUSION_FLICKR_X4,
        "table6_complexity_x8": TABLE6_COMPLEXITY,
    }
