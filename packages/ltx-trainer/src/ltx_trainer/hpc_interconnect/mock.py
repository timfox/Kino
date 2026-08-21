"""evaluation_smoke for HPC interconnect congestion stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hpc_interconnect.benchmarks import FIG5_STEADY_ANCHORS, PAPER_ARXIV, benchmarks_bundle
from ltx_trainer.hpc_interconnect.experiment import run_congestion_experiment, sweep_steady_heatmap
from ltx_trainer.hpc_interconnect.fabrics import AggressorPattern, SystemName
from ltx_trainer.hpc_interconnect.simulator import predict_performance_ratio


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "hpc_interconnect",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
    }
    try:
        leo_incast = predict_performance_ratio(
            SystemName.LEONARDO,
            nodes=64,
            aggressor=AggressorPattern.INCAST,
            message_bytes=32 * 1024,
        )
        cresco_a2a = predict_performance_ratio(
            SystemName.CRESCO8,
            nodes=64,
            aggressor=AggressorPattern.ALLTOALL,
        )
        lumi_incast = predict_performance_ratio(
            SystemName.LUMI,
            nodes=256,
            aggressor=AggressorPattern.INCAST,
        )

        exp_leo = run_congestion_experiment(
            SystemName.LEONARDO,
            nodes=64,
            aggressor=AggressorPattern.INCAST,
            iterations=120,
        )
        exp_lumi = run_congestion_experiment(
            SystemName.LUMI,
            nodes=64,
            aggressor=AggressorPattern.ALLTOALL,
            iterations=120,
        )
        bursty_cresco = run_congestion_experiment(
            SystemName.CRESCO8,
            nodes=64,
            aggressor=AggressorPattern.INCAST,
            steady=False,
            burst_pause_collectives=1,
            iterations=120,
        )
        heatmap_rows = len(
            sweep_steady_heatmap(SystemName.CRESCO8, node_counts=[32, 64], aggressor=AggressorPattern.ALLTOALL)
        )

        nanjing_nslb_on = predict_performance_ratio(
            SystemName.NANJING,
            nodes=8,
            aggressor=AggressorPattern.ALLTOALL,
            nslb_enabled=True,
        )
        nanjing_nslb_off = predict_performance_ratio(
            SystemName.NANJING,
            nodes=8,
            aggressor=AggressorPattern.ALLTOALL,
            nslb_enabled=False,
        )

        out.update(
            {
                "torch": False,
                "predicted_leonardo_incast_64": round(leo_incast, 3),
                "predicted_cresco8_alltoall_64": round(cresco_a2a, 3),
                "predicted_lumi_incast_256": round(lumi_incast, 3),
                "ref_leonardo_incast_min": FIG5_STEADY_ANCHORS["leonardo_incast_64"]["min_ratio"],
                "experiment_leonardo_incast_ratio": exp_leo["predicted_performance_ratio"],
                "experiment_lumi_alltoall_ratio": exp_lumi["predicted_performance_ratio"],
                "bursty_cresco_incast_ratio": bursty_cresco["predicted_performance_ratio"],
                "heatmap_sample_rows": heatmap_rows,
                "nanjing_nslb_on": round(nanjing_nslb_on, 3),
                "nanjing_nslb_off": round(nanjing_nslb_off, 3),
                "ordering_ok": lumi_incast > leo_incast and cresco_a2a < lumi_incast,
            }
        )
    except Exception as exc:  # pragma: no cover
        out["error"] = str(exc)
    return out
