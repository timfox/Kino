"""CounterFlow: two-phase inference for counterfactual video foley (arXiv:2605.18916)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CounterFlowConfig:
    paper_arxiv: str = "arXiv:2605.18916"
    demo_url: str = "https://gyubin-lee.github.io/counterflow-demo/"
    backbone: str = "MMAudio large 44k v2"
    num_steps: int = 25
    transition_step: int = 17  # N_trans
    guidance_w_vid: float = 3.0
    guidance_w_txt: float = 5.0
    guidance_w_cfg: float = 4.5
    output_seconds: float = 8.0
    dataset: str = "VGGSound-Sparse Clean"
    dataset_test_videos: int = 451
    dataset_source_captions: int = 12
    dataset_triplets: int = 4961
