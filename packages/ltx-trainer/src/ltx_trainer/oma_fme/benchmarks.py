"""Reference results (Premkumar & Herglotz, MHV '26 / arXiv:2601.17568)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.oma_fme.config import PAPER_ARXIV, PAPER_DOI, PAPER_TITLE, PAPER_URL

# Table 3 excerpts — averages and headline CMP-PRA HQ row
TABLE3_ERP_CRC_AVG: dict[str, Any] = {
    "method": "ERP-CRC",
    "anchor": "avg",
    "BD_PSNR_db": 0.02,
    "BD_WSPSNR_db": 0.00,
    "BDET_pct": -42.39,
    "delta_T_S_pct": -42.08,
    "delta_T_P_pct": -40.56,
}

TABLE3_CMP_PRA_HQ_AVG: dict[str, Any] = {
    "method": "CMP-PRA",
    "anchor": "HQ",
    "BD_PSNR_db": -1.08,
    "BD_WSPSNR_db": -0.77,
    "BDET_pct": -50.22,
    "delta_T_S_pct": -51.38,
    "delta_T_P_pct": -49.88,
}

TABLE3_CMP_CRC_HQ_AVG: dict[str, Any] = {
    "method": "CMP-CRC",
    "anchor": "HQ",
    "BD_PSNR_db": -1.07,
    "BD_WSPSNR_db": -0.76,
    "delta_T_S_pct": -50.82,
    "delta_T_P_pct": -48.67,
}

# Sec. 5 headline ranges
HEADLINE: dict[str, Any] = {
    "erp_serial_speedup_pct_range": (33, 59),
    "erp_parallel_speedup_pct_range": (24, 57),
    "cmp_avg_serial_speedup_pct": 51.38,
    "cmp_avg_parallel_speedup_pct": 49.88,
    "max_wall_clock_speedup_x": 4.2,
    "bdet_pct_up_to": -50.0,
}

# Table 2 experimental parameters
TABLE2_PARAMS: dict[str, Any] = {
    "resolutions": ["2048x1024 (HD)", "4096x2048 (4K)", "8192x4096 (8K)"],
    "QPs": [22, 27, 32, 37, 42],
    "encoder": "x265 medium, intra period 1s, 4 threads",
    "decoder": "HM 1 thread",
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "doi": PAPER_DOI,
        "paper_url": PAPER_URL,
        "table2_params": TABLE2_PARAMS,
        "table3_erp_crc_avg": TABLE3_ERP_CRC_AVG,
        "table3_cmp_crc_hq_avg": TABLE3_CMP_CRC_HQ_AVG,
        "table3_cmp_pra_hq_avg": TABLE3_CMP_PRA_HQ_AVG,
        "headline": HEADLINE,
    }
