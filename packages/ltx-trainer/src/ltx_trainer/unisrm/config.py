"""UniSRM unified speech reward model stub (arXiv:2605.23261)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UnisrmConfig:
    paper_arxiv: str = "arXiv:2605.23261"
    title: str = "UniSRM: unified speech reward model with reasoning-based fine-grained assessment"
    backbone: str = "Qwen2.5-Omni-7B-thinker"
    repo_url: str = "https://github.com/lavendery/UniSRM"
    grpo_group_size: int = 8
    kl_coefficient: float = 0.04
    reward_weight_fmt: float = 1.0
    reward_weight_acc: float = 1.0
    reward_weight_rc: float = 1.0
    # Table 1 — UNISRM-BENCH accuracy (%) / T2 PCC
    bench_t1_acc: float = 65.06
    bench_t2_acc: float = 39.74
    bench_t2_pcc: float = 0.551
    bench_t3_en_acc: float = 85.61
    bench_t3_zh_acc: float = 91.30
    bench_t4_acc: float = 88.89
    ablation_wo_grpo_t1: float = 60.24
    ablation_wo_rcr_t1: float = 60.44
    dataset_total_samples: int = 46_259
