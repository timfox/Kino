"""Dense360 — omnidirectional dense VLM understanding (arXiv:2506.14471)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2506.14471"
PAPER_TITLE = "Dense360: Dense Understanding from Omnidirectional Panoramas"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

PANORAMA_COUNT = 160_000
ENTITY_CAPTIONS = 5_000_000
REFERRING_EXPRESSIONS = 1_000_000
SCENE_DESCRIPTIONS = 100_000
BENCH_ERP_IMAGES = 1_279
BENCH_ENTITIES = 3_000
SEG_TOKEN = "[SEG]"
BASELINE_VLM = "Qwen2.5-VL-3B-Instruct"


@dataclass
class Dense360Config:
    height: int = 256
    width: int = 512
    feature_dim: int = 64
    use_erp_rope: bool = True
    patch_size: int = 16
