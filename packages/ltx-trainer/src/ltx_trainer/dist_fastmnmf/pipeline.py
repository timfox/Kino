"""Framework card and APSIPA ASC benchmark excerpts (arXiv:2605.19388)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dist_fastmnmf.complexity import complexity_table_rows
from ltx_trainer.dist_fastmnmf.config import DistFastmnmfConfig, SdrResults, TimingResults
from ltx_trainer.dist_fastmnmf.layout import LIMITATIONS
from ltx_trainer.dist_fastmnmf.mock import evaluation_smoke


def sdr_results() -> SdrResults:
    """§IV-B mean SDR improvement (dB) — Fig. 2."""
    return SdrResults(
        three_source={
            "distributed_fastmnmf": 13.4,
            "fastmnmf_one_subarray": 12.5,
            "fastmnmf_all_subarrays": 15.7,
        },
        five_source={
            "distributed_fastmnmf": 6.3,
            "fastmnmf_one_subarray": 5.8,
            "fastmnmf_all_subarrays": 7.3,
        },
    )


def table_ii_computation_time() -> list[dict[str, Any]]:
    """Table II — three-source, 10 s mixture, 200 iterations."""
    t = TimingResults()
    return [
        {"method": "FastMNMF (one subarray)", "time_s": t.fastmnmf_one_subarray_s, "se": 0.3},
        {"method": "FastMNMF (all subarrays)", "time_s": t.fastmnmf_all_subarrays_s, "se": 0.7},
        {"method": "Distributed FastMNMF", "time_s": t.distributed_fastmnmf_s, "se": 2.4},
    ]


def experiment_setup() -> dict[str, Any]:
    cfg = DistFastmnmfConfig()
    return {
        "room_m": cfg.room_m,
        "rt60_ms": cfg.rt60_ms,
        "subarrays": cfg.n_subarrays_l,
        "mics_per_subarray": cfg.mics_per_subarray,
        "total_mics": cfg.total_mics_m,
        "sources_tested": [3, 5],
        "local_underdetermined_note": "5 sources with 4 mics/subarray — locally underdetermined, globally overdetermined",
        "mixtures": cfg.n_mixtures,
        "nmf_inits": cfg.n_nmf_inits,
        "K": cfg.nmf_bases_k,
        "iterations": cfg.n_iterations,
        "corpus": "JNAS speech",
    }


def framework_card(cfg: DistFastmnmfConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DistFastmnmfConfig()
    sdr = sdr_results()
    return {
        "name": cfg.title,
        "paper": f"arXiv:{cfg.paper_id}",
        "paper_url": cfg.paper_url,
        "venue": cfg.venue,
        "authors": "Hirotaka Nishikori, Nobutaka Ito, Kouei Yamaoka, Norihiro Takamune, Hiroshi Saruwatari (UTokyo / AIST)",
        "problem": (
            "Distributed microphone arrays need BSS over wide areas, but joint FastMNMF on all mics "
            "requires O(M^4) IP inversions per frequency; single-subarray FastMNMF is cheaper but ignores other subarrays."
        ),
        "method": {
            "base": "FastMNMF — jointly diagonalizable SCMs + NMF spectrograms (Eqs. 2–4)",
            "constraint": "Block-diagonal R_in across subarrays (Eq. 11); per-subarray joint diagonalization (Eq. 12)",
            "shared": "NMF source spectrograms h_ijn shared across subarrays; inter-subarray covariance discarded",
            "updates": "Per-subarray IP (14)–(16) + global (8)–(10) for t, v, Λ^(l)",
        },
        "goal": "Computationally efficient middle ground between FastMNMF (all) and FastMNMF (one subarray)",
        "headlines": headline_results(cfg),
        "sdr_three_source": sdr.three_source,
        "sdr_five_source": sdr.five_source,
        "limitations": LIMITATIONS,
    }


def headline_results(cfg: DistFastmnmfConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DistFastmnmfConfig()
    sdr = sdr_results()
    t = TimingResults()
    return {
        "sdr_gain_vs_one_subarray_3src_db": 0.8,
        "sdr_gain_vs_one_subarray_5src_db": round(sdr.five_source["distributed_fastmnmf"] - sdr.five_source["fastmnmf_one_subarray"], 1),
        "runtime_vs_all_subarrays_fraction": round(t.distributed_fastmnmf_s / t.fastmnmf_all_subarrays_s, 3),
        "speedup_vs_all_subarrays_x": round(t.fastmnmf_all_subarrays_s / t.distributed_fastmnmf_s, 2),
        "locally_underdetermined_5src_applicable": True,
    }


def evaluation_demo(cfg: DistFastmnmfConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DistFastmnmfConfig()
    return {"paper": f"arXiv:{cfg.paper_id}", "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: DistFastmnmfConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DistFastmnmfConfig()
    return {
        "framework": framework_card(cfg),
        "table1_complexity": complexity_table_rows(),
        "table2_timing": table_ii_computation_time(),
        "experiment_setup": experiment_setup(),
        "sdr_results": {
            "three_source": sdr_results().three_source,
            "five_source": sdr_results().five_source,
        },
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }
