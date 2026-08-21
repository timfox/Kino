"""Paper table excerpts for CoRe-KD (arXiv:2605.29590)."""

from __future__ import annotations

from typing import Any

# Table 1 — IEMOCAP-6 / MELD-7 fixed missing (CoRe-KD row, selected conditions)
TABLE1_IEMOCAP6_CORE_KD = {
    "{a}": (57.86, 57.82),
    "{v}": (40.36, 36.36),
    "{l}": (67.53, 67.56),
    "{l,a,v}": (74.68, 74.74),
}

TABLE1_MELD7_CORE_KD = {
    "{a}": (49.77, 41.08),
    "{v}": (48.35, 32.24),
    "{l}": (68.28, 67.25),
    "{l,a,v}": (68.85, 67.41),
}

# Table 1 — random missing IEMOCAP-6 @ rate 0.7
TABLE1_IEMOCAP6_RM_07 = (68.45, 68.30)

# Table 3 ablation — IEMOCAP-6 full vs w/o CSA
TABLE3_IEMOCAP6 = {
    "CoRe-KD": {"fixed_lav": (74.68, 74.74), "rm_07": (68.45, 68.30)},
    "w/o LCSA": {"fixed_lav": (72.89, 73.12), "rm_07": (63.96, 64.31)},
    "w/o LNCE": {"fixed_lav": (73.81, 73.88), "rm_07": (66.48, 66.70)},
}

# Mechanism — state drift reduction & rejection rate (Fig. 3)
MECHANISM = {
    "state_drift_reduction_pct": 92.6,
    "rejection_rate_baseline_pct": 69.3,
    "rejection_rate_core_kd_pct": 96.5,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_iemocap6": TABLE1_IEMOCAP6_CORE_KD,
        "table1_meld7": TABLE1_MELD7_CORE_KD,
        "table1_iemocap6_rm07": TABLE1_IEMOCAP6_RM_07,
        "table3_ablation": TABLE3_IEMOCAP6,
        "mechanism": MECHANISM,
        "beats_sota_iemocap6_lav": True,
        "datasets": ["IEMOCAP-4", "IEMOCAP-6", "MELD-7", "CMU-MOSEI"],
    }
