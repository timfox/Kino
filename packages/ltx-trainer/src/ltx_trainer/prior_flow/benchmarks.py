"""Paper tables (Liu et al., arXiv:2506.23897)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.prior_flow.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 6 — MPF EFT/City + FlowScape (PriOr-RAFT row, EPE / SEPE)
TABLE6_SOTA = {
    "SphereNet_RAFT": {"mpf_eft_epe": 13.2, "mpf_city_epe": 8.28, "flowscape_all_epe": 12.9},
    "TanImg_RAFT": {"mpf_eft_epe": 4.38, "mpf_city_epe": 3.13, "flowscape_all_epe": 18.3},
    "SLOF_RAFT": {"mpf_eft_epe": 4.98, "mpf_city_epe": 1.35, "flowscape_all_epe": 7.59},
    "PanoFlow_RAFT": {"flowscape_all_epe": 3.38, "flowscape_all_sepe": 4.78},
    "PriOr_RAFT": {
        "mpf_eft_epe": 3.30,
        "mpf_eft_sepe": 6.23,
        "mpf_city_epe": 1.13,
        "mpf_city_sepe": 1.88,
        "mpf_all_epe": 2.22,
        "mpf_all_sepe": 4.06,
        "flowscape_sunny_epe": 2.41,
        "flowscape_all_epe": 2.33,
        "flowscape_all_sepe": 3.49,
    },
}

# Table 1 — module ablation (EFT, EPE / SEPE)
TABLE1_ABLATION = {
    "RAFT_baseline": {"equator_epe": 1.09, "poles_epe": 7.90, "all_epe": 4.49, "all_sepe": 7.43},
    "Ortho_DCCL": {"equator_epe": 1.08, "poles_epe": 7.56, "all_epe": 4.32, "all_sepe": 7.20},
    "PriOr_RAFT_full": {"equator_epe": 1.03, "poles_epe": 5.57, "all_epe": 3.30, "all_sepe": 6.23},
}

# Table 3 — universality on MPF EFT
TABLE3_UNIVERSALITY = {
    "RAFT": {"epe": 4.49, "sepe": 7.43},
    "PriOr_RAFT_4iter": {"epe": 3.89, "sepe": 7.17},
    "PriOr_RAFT": {"epe": 3.30, "sepe": 6.23},
    "GMA": {"epe": 4.26, "sepe": 7.07},
    "PriOr_GMA": {"epe": 3.25, "sepe": 6.17},
    "SKFlow": {"epe": 3.79, "sepe": 6.55},
    "PriOr_SKFlow": {"epe": 3.19, "sepe": 6.13},
}

# Table 7 — FlowScape regions vs PanoFlow
TABLE7_REGIONS = {
    "PanoFlow": {"equator_epe": 0.52, "poles_epe": 6.25, "all_epe": 3.38},
    "PriOr_RAFT": {"equator_epe": 0.53, "poles_epe": 4.13, "all_epe": 2.33},
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table6_sota": TABLE6_SOTA,
        "table1_ablation": TABLE1_ABLATION,
        "table3_universality": TABLE3_UNIVERSALITY,
        "table7_regions": TABLE7_REGIONS,
    }
