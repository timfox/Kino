"""Holistic taxonomy of energy-aware computing (Fig. 2, Table 1)."""

from __future__ import annotations

from typing import Any

# Survey pillars → representative topics (not exhaustive bibliography).
TAXONOMY: dict[str, list[str]] = {
    "profiling_methodology": [
        "in-band RAPL/APM",
        "out-of-band HDEEM/TSDB",
        "ML power prediction (DeepPM, EffiCast)",
        "MLPerf Power",
        "observer effect / eBPF (Kepler)",
    ],
    "hardware": [
        "edge IoT milliwatt SWaP",
        "ARM vs x86 cloud",
        "AI super-accelerators (B200, MI300X, TPU)",
        "CGRA approximate edge",
        "photonic / PIM / chiplet UCIe",
    ],
    "software_optimization": [
        "energy-aware compilation (MLIR, SYCL)",
        "MPI COUNTDOWN idle C-states",
        "GreenLA / framework efficiency",
        "energy smells in ML code",
    ],
    "dynamic_power_management": [
        "DVFS / Q-learning schedulers",
        "SLURM power caps (Variorum)",
        "phase-aware core/uncore",
        "GPU frequency capping",
    ],
    "scheduling_placement": [
        "VM consolidation / bin packing",
        "carbon-aware temporal/spatial shifting",
        "federated learning schedulers",
        "PSO / DRL orchestration",
    ],
    "green_ai": [
        "red vs green AI",
        "carbon elasticity training",
        "prefill vs decode inference DVFS",
        "RAG KV-cache carbon",
        "quantization / MoE / distillation",
    ],
    "cooling_facility": [
        "liquid cold plate D2C",
        "two-phase immersion (PFAS limits)",
        "WUE / AWI water stress",
        "heat reuse (ERF)",
        "joint thermal-workload DRL",
    ],
}

CORPUS_KEYWORD_TOP: list[tuple[str, int]] = [
    ("energy efficiency", 99),
    ("energy consumption", 77),
    ("sustainability", 67),
    ("carbon footprint", 61),
    ("cloud computing", 46),
    ("green computing", 41),
    ("power demand", 27),
    ("energy-aware scheduling", 26),
    ("data centers", 26),
    ("power management", 25),
]


def taxonomy_dict() -> dict[str, Any]:
    return {
        "pillars": {k: list(v) for k, v in TAXONOMY.items()},
        "corpus_keyword_top": list(CORPUS_KEYWORD_TOP),
        "continuum": ["edge/IoT", "cloud", "HPC/exascale"],
    }
