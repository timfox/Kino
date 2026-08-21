"""MetaphorVU configuration (arXiv:2605.25488)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetaphorVUConfig:
    """Reference hyperparameters from MetaphorVU / MetaphorBoost."""

    paper_arxiv: str = "arXiv:2605.25461"
    project_page: str = "https://github.com/icip-cas/MetaphorVU"
    benchmark_hf: str = "https://huggingface.co/datasets/lzq2021/MetaphorVU-Bench"
    num_videos: int = 860
    kg_nodes: int = 54687
    kg_edges: int = 200268
    query_hops: int = 2
    top_z: int = 10
    judge_model: str = "DeepSeek-V3.2"
    human_upper_bound_avg: float = 83.4
    taxonomy: tuple[str, ...] = (
        "Body Language",
        "Atmosphere Language",
        "Cultural Symbol",
        "Naturalistic Symbol",
        "Causal Montage",
        "Analogical Montage",
        "Surreal Narrative",
        "Performative Narrative",
    )
