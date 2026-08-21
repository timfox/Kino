"""Paper card export."""

from __future__ import annotations

from typing import Any

from ltx_trainer.genpcr.benchmarks import benchmarks_bundle
from ltx_trainer.genpcr.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.genpcr.datasets import datasets_card
from ltx_trainer.genpcr.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "authors": [
            "Haobo Jiang",
            "Jin Xie",
            "Jian Yang",
            "Liang Yu",
            "Jianmin Zheng",
        ],
        "affiliation": "NTU / Nankai / Alibaba / NJU",
        "components": [
            "DepthMatch-ControlNet (depth → perspective RGB pairs)",
            "LiDARMatch-ControlNet (ERP range map → panoramic RGB)",
            "Coupled conditional denoising + coupled prompts",
            "Zero-shot geometric-color fusion (DINOv2 / SD) + XYZ-RGB",
        ],
        "contributions": [
            "Generative PCR: synthesize aligned RGB for geometry-only registration",
            "Plug-and-play on FCGF, Predator, GeoTrans, ColorPCR",
            "ScanNet/3DMatch + Dur360BEV SOTA gains with ~13M params",
        ],
        "datasets": datasets_card(),
        "benchmarks": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "genpcr", **evaluation_demo_run()}
