"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.set_cuda_graph.benchmarks import (
    fig6_overhead_anchors,
    summary_anchors,
    table_1_averages,
    table_1_speedups,
    table_2_overhead,
)
from ltx_trainer.set_cuda_graph.constants import (
    BASELINE_MODELS,
    CUDA_VERSION,
    HARDWARE,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
)
from ltx_trainer.set_cuda_graph.overhead import overhead_model_card
from ltx_trainer.set_cuda_graph.references import reference_anchors
from ltx_trainer.set_cuda_graph.scheduling import algorithms_summary, runtime_components_card
from ltx_trainer.set_cuda_graph.workloads import workload_catalog


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "authors": "Zhengxiong Li, Tsung-Wei Huang, Umit Ogras",
        "affiliation": "University of Wisconsin–Madison",
        "cuda_version": CUDA_VERSION,
        "findings": [
            "Event-chained host–device co-scheduling for CUDA graph pipelines",
            "Per-worker queues + work-stealing with O(1) callback synchronization",
            "1.15–1.44× throughput vs batching/queue baselines on six workloads",
            "18–54% lower scheduling overhead than SOTA host-side models",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "hardware": HARDWARE,
        "baselines": list(BASELINE_MODELS),
        "workloads": workload_catalog(),
        "overhead_model": overhead_model_card(),
        "runtime": runtime_components_card(),
        "algorithms": algorithms_summary(),
        "table_1": table_1_speedups(),
        "table_1_averages": table_1_averages(),
        "table_2": table_2_overhead(),
        "fig_6": fig6_overhead_anchors(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
