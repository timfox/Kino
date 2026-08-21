"""Plug-in EDL stub (arXiv:2605.22746)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlugEdlConfig:
    paper_arxiv: str = "arXiv:2605.22746"
    title: str = "Plug-in losses for evidential deep learning"
    num_classes: int = 30  # GSC v1 full task
    evidence_prior: float = 1.0  # classical α_i = e_i + 1
    kl_ramp_epochs: int = 400
    # Headline GSC v1 results (Table 2, Softmax, entropy @ 99.9% Accth)
    softmax_base_acc_pct: float = 97.21
    softmax_entropy_acctotal_99_9_pct: float = 88.41
    plug_ce_base_acc_pct: float = 96.84
