"""Table 1–4 and user-study anchors."""

from __future__ import annotations

TABLE1 = {
    "UDIS-D": {
        "Ours": {"NiQE": 4.169, "BRISQUE": 37.57, "PIQE": 22.60, "PSQ": 0.10},
        "UDIS++": {"NiQE": 4.209, "BRISQUE": 37.84, "PIQE": 22.97, "PSQ": 0.17},
    },
    "DAVISProcessed10": {
        "Ours": {"NiQE": 3.296, "BRISQUE": 20.77, "PIQE": 15.03, "PSQ": 0.11},
        "UDIS++": {"NiQE": 3.448, "BRISQUE": 20.14, "PIQE": 13.75, "PSQ": 0.14},
    },
    "RealWorld400": {
        "Ours": {"NiQE": 3.188, "BRISQUE": 26.48, "PIQE": 11.75, "PSQ": 0.09},
        "UDIS++": {"NiQE": 3.312, "BRISQUE": 26.37, "PIQE": 11.98, "PSQ": 0.11},
    },
}

TABLE2_USER_STUDY = {
    "Coherence": {"Ours": 4.8, "UDIS++": 4.2, "GC": 3.8},
    "Integrity": {"Ours": 4.9, "UDIS++": 4.1, "GC": 3.7},
    "Quality": {"Ours": 4.8, "UDIS++": 4.1, "GC": 3.6},
}

TABLE3_EFFICIENCY = {
    "UDIS++": {"params_M": 33.56, "GFLOPs": 80.46},
    "Ours": {"params_M": 33.55, "GFLOPs": 10.77},
}

TABLE4_ABLATION = {
    "w/o Dynamic Mask Opt.": {"Failure_Cases": 64, "Rate": 0.84},
    "w/o Exclusivity Loss": {"Failure_Cases": 132, "Rate": 0.67},
    "Ours": {"Failure_Cases": 52, "Rate": 0.87},
}


def benchmarks_bundle() -> dict:
    return {
        "table1_quantitative": TABLE1,
        "table2_user_study": TABLE2_USER_STUDY,
        "table3_efficiency": TABLE3_EFFICIENCY,
        "table4_ablation": TABLE4_ABLATION,
    }
