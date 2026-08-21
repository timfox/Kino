"""MAS-IB — when multi-agent systems help via information bottleneck (arXiv:2607.16133)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2607.16133"
PAPER_TITLE = "When Do Multi-Agent Systems Help? An Information Bottleneck Perspective"
PAPER_SYSTEM = "MAS-IB"
PAPER_AUTHORS = (
    "Wendi Yu, Lianhao Zhou, Xiangjue Dong, Sai Sudarshan Barath, Declan Staunton, "
    "Byung-Jun Yoon, Xiaoning Qian, James Caverlee, Shuiwang Ji (Texas A&M / BNL)"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_GITHUB = "https://github.com/divelab/MAS-SAS"
BENCHMARK = "ALFWorld / WebShop / WorkBench / WideSearch / TravelPlanner × Qwen2.5-7B / GPT-4o-mini / Qwen3.5-27B"

COMPONENTS = (
    "mas_sas_equivalence",
    "relay_information_bottleneck",
    "mas_gain_decomposition",
    "controlled_prototypes",
)

PROTOTYPES = ("SAS", "SAS-contextflow", "MAS")
MODELS = ("Qwen2.5-7B", "GPT-4o-mini", "Qwen3.5-27B")


@dataclass
class MasIbConfig:
    """Runtime knobs for the CPU stub."""

    beta: float = 1.0  # effective capability (↑ with stronger LLMs)
    n_workers: int = 3
    relay_bits: float = 8.0  # compressed relay description length proxy
    full_context_bits: float = 64.0
    delta_loss: float = 0.05  # relay information loss Δ_i(m)
    enable_compression: bool = True
