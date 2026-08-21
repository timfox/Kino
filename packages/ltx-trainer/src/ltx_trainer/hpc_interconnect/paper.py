"""HPC interconnect congestion paper stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hpc_interconnect.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "authors": "Piarulli, Faltelli, Pleiter, Sivalingam, Zhang, Zhao, Turisini, Iannone, Artigiani, De Sensi",
        "problem": (
            "Multi-tenant HPC/AI jobs share interconnects; congestion from AlltoAll vs Incast "
            "patterns degrades MPI collectives differently across IB, Slingshot, and RoCE/NSLB Ethernet."
        ),
        "method": {
            "platforms": list(b["table1_systems"].keys()),
            "victim": "Custom ring AllGather / linear AlltoAll (MPI send/recv, no library algos)",
            "aggressors": "Continuous AlltoAll or Incast noise; interleaved node allocation",
            "modes": "Steady persistent congestion + bursty (variable burst length and idle gap)",
            "metric": "Mean runtime ratio uncongested/congested after 100-iteration warmup",
        },
        "key_results": {
            "leonardo_incast_64": "ratio can drop to ~0.2 (5× slowdown)",
            "cresco8_alltoall_64": "ratio down to ~0.45 at scale",
            "lumi_256": "within ~5% of baseline for AlltoAll and Incast",
            "nanjing_nslb": "NSLB on: no drop; off: 180→120 Gb/s",
        },
        "observations": b["observations"],
        "reference_metrics": b,
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.hpc_interconnect.mock import evaluation_smoke

    return evaluation_smoke()
