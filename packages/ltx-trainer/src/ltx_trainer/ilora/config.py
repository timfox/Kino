"""iLoRA configuration (arXiv:2605.30179)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ILORAConfig:
    paper_arxiv: str = "arXiv:2605.30179"
    num_taxa: int = 20  # K selected entities (MaAsLin2 top-20 for IBD)
    lora_rank: int = 16
    lora_alpha: int = 32
    graph_dim: int = 128
    lambda_pois: float = 1e-3
    lambda_lap: float = 1e-3
    laplace_prior_scale: float = 1.0
    mc_graph_samples: int = 1
    ibd_backbone: str = "Qwen3-8B"
    molweni_backbone: str = "Llama-3.1-8B-Instruct"
    ltx_lora_rank: int = 16  # bridge default for DiT PEFT
    ltx_target_hidden: int = 4096
