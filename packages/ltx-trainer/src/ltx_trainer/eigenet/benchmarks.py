"""Paper table excerpts — AcousticRooms (AR) and HAA (Sec. IV)."""

from __future__ import annotations

from typing import Any

# Table I — DAC reconstruction on AR
TABLE1_DAC_AR = {
    "EDT": 0.004,
    "C50": 0.606,
    "T60": 4.963,
}

# Table II — AR test, K ∈ {1,4,8} — EIGENET row highlights
TABLE2_AR_EIGENET: dict[int, dict[str, float]] = {
    1: {"EDT": 0.052, "C50": 1.488, "T60": 10.213},
    4: {"EDT": 0.047, "C50": 1.398, "T60": 8.061},
    8: {"EDT": 0.041, "C50": 1.242, "T60": 7.605},
}

TABLE2_AR_XRIR: dict[int, dict[str, float]] = {
    1: {"EDT": 0.076, "C50": 2.124, "T60": 12.617},
    4: {"EDT": 0.054, "C50": 1.540, "T60": 10.052},
    8: {"EDT": 0.050, "C50": 1.442, "T60": 9.393},
}

# Table III — HAA dampenedBase T60 (%) excerpt
TABLE3_HAA_DAMPENED_T60: dict[str, dict[int, float]] = {
    "xRIR": {1: 193.582, 4: 191.347, 8: 191.392},
    "Ours": {1: 49.090, 4: 46.072, 8: 49.261},
}

# Table V — modulation target ablation (K=8 excerpt)
TABLE5_MODULATION_K8 = {
    "AA w/o mod": {"EDT": 0.048, "C50": 1.374, "T60": 8.443},
    "Var.1": {"EDT": 0.044, "C50": 1.357, "T60": 8.512},
    "Var.2": {"EDT": 0.043, "C50": 1.260, "T60": 9.902},
    "Ours": {"EDT": 0.041, "C50": 1.242, "T60": 7.605},
}

# Table VI — modulation generalizes across attention (K=8, AA w/ mod = full model)
TABLE6_ATTENTION_MOD_K8 = {
    "CA w/o mod": {"EDT": 0.104, "C50": 3.332, "T60": 9.677},
    "CA w/ mod": {"EDT": 0.044, "C50": 1.302, "T60": 8.513},
    "SA w/o mod": {"EDT": 0.047, "C50": 1.476, "T60": 8.731},
    "SA w/ mod": {"EDT": 0.042, "C50": 1.283, "T60": 8.902},
    "AA w/o mod": {"EDT": 0.048, "C50": 1.374, "T60": 8.443},
    "AA w/ mod": {"EDT": 0.041, "C50": 1.242, "T60": 7.605},
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_dac_ar": TABLE1_DAC_AR,
        "table2_ar_eigenet": TABLE2_AR_EIGENET,
        "table2_ar_xrir": TABLE2_AR_XRIR,
        "table3_haa_dampened_t60": TABLE3_HAA_DAMPENED_T60,
        "table5_modulation_k8": TABLE5_MODULATION_K8,
        "table6_attention_mod_k8": TABLE6_ATTENTION_MOD_K8,
    }
