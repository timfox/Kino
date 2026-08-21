"""Online min-cost matching with general arrivals (arXiv:2606.05546)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OmcmGeneralConfig:
    paper_arxiv: str = "arXiv:2606.05546"
    title: str = "Online Min-Cost Matching with General Arrivals"
    unknown_iid_cr: str = "O(log² n)"
    random_order_cr: str = "unbounded (lower bound)"
    line_metric: bool = True
