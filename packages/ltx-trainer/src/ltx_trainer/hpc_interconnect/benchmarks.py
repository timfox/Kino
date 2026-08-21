"""Reference metrics from Piarulli et al. (arXiv:2604.11432)."""

from __future__ import annotations

from typing import Any

PAPER_ARXIV = "2604.11432"
PAPER_TITLE = "Characterizing the Impact of Congestion in Modern HPC Interconnects"

# Table I — evaluated systems (excerpt)
TABLE1_SYSTEMS: dict[str, dict[str, str | int]] = {
    "leonardo": {
        "site": "CINECA",
        "nodes": 3456,
        "interconnect": "HDR InfiniBand",
        "topology": "Dragonfly+",
        "link_per_node": "400 Gb/s",
    },
    "cresco8": {
        "site": "ENEA",
        "nodes": 760,
        "interconnect": "NDR InfiniBand",
        "topology": "1.67:1 blocking Fat-Tree",
        "link_per_node": "200 Gb/s",
    },
    "lumi": {
        "site": "CSC",
        "nodes": 2978,
        "interconnect": "Cray Slingshot",
        "topology": "Dragonfly",
        "link_per_node": "800 Gb/s (4×200)",
    },
    "haicgu": {
        "site": "Goethe / OEHI",
        "nodes": 10,
        "interconnect": "EDR IB / RoCE CE8850",
        "topology": "single switch",
        "link_per_node": "100 GE",
    },
    "nanjing": {
        "site": "Huawei lab",
        "nodes": 8,
        "interconnect": "RoCE NSLB CE9855",
        "topology": "2-leaf 2-spine",
        "link_per_node": "200 GE",
    },
}

# Fig. 5 — steady congestion ratio anchors (uncongested / congested runtime, higher is better)
FIG5_STEADY_ANCHORS: dict[str, dict[str, float]] = {
    "cresco8_alltoall_64": {"min_ratio": 0.45, "typical": 0.70},
    "cresco8_incast_64": {"min_ratio": 0.60, "typical": 0.60},
    "leonardo_alltoall_256": {"min_ratio": 0.82, "typical": 0.98},
    "leonardo_incast_64": {"min_ratio": 0.20, "typical": 0.20},
    "lumi_alltoall_256": {"min_ratio": 0.95, "typical": 1.00},
    "lumi_incast_256": {"min_ratio": 0.95, "typical": 1.00},
}

# Fig. 4 — Nanjing NSLB (4v+4 nodes AlltoAll victim peak Gb/s)
FIG4_NANJING_NSLB: dict[str, float] = {
    "peak_gbps_uncongested": 180.0,
    "peak_gbps_congested_nslb_on": 180.0,
    "peak_gbps_congested_nslb_off": 120.0,
}

# Fig. 6 — bursty 64-node AllGather victim ratio anchors
FIG6_BURSTY_64: dict[str, dict[str, float]] = {
    "cresco8_alltoall": {"typical_ratio": 0.70},
    "cresco8_incast": {"typical_ratio": 0.08},
    "leonardo_alltoall": {"typical_ratio": 0.95},
    "leonardo_incast_short_gap": {"typical_ratio": 0.15},
    "lumi_alltoall": {"typical_ratio": 0.98},
    "lumi_incast": {"typical_ratio": 0.98},
}

METHODOLOGY = {
    "victim_collectives": ("allgather_ring", "alltoall_linear"),
    "aggressor_collectives": ("alltoall", "incast"),
    "warmup_iterations": 100,
    "measured_iterations": 900,
    "max_reported_nodes": 256,
    "victim_aggressor_interleave": True,
}

OBSERVATIONS = [
    "High offered load can trigger CC instability without foreign jobs (CE8850 sawtooth).",
    "CRESCO8 AlltoAll degrades at scale; Leonardo Incast collapses at edge (~5× at 64 nodes).",
    "LUMI Slingshot maintains near-baseline under steady and bursty Incast/AlltoAll.",
    "Bursty traffic with short idle gaps prevents queue drain and CC convergence.",
    "Topology alone does not predict resilience; fabric generation + CC/AR tuning dominate.",
]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_systems": TABLE1_SYSTEMS,
        "fig5_steady_anchors": FIG5_STEADY_ANCHORS,
        "fig4_nanjing_nslb": FIG4_NANJING_NSLB,
        "fig6_bursty_64": FIG6_BURSTY_64,
        "methodology": METHODOLOGY,
        "observations": OBSERVATIONS,
    }
