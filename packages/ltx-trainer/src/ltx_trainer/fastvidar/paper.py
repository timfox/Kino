"""Paper card."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fastvidar.benchmarks import benchmarks_bundle
from ltx_trainer.fastvidar.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL
from ltx_trainer.fastvidar.datasets import datasets_card
from ltx_trainer.fastvidar.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "authors": [
            "Hangtian Zhao",
            "Xiang Chen",
            "Yizhe Li",
            "Qianhao Wang",
            "Haibo Lu",
            "Fei Gao",
        ],
        "affiliation": "USTC / ECNU / Xidian / ZJU FAST Lab",
        "components": [
            "ERP fisheye unification (Eq. 1–2)",
            "Alternative Hierarchical Attention: window + frame + global (AHA)",
            "ERP mean fusion for 360° depth (Eq. 14–15)",
            "ERP latitude-weighted Huber + gradient loss (Eq. 16–18)",
        ],
        "contributions": [
            "AHA: <10% overhead vs window-only; ~16× vs full attention (theory)",
            "3.3× faster than VGGT @ 640×320×4; 20 FPS on Orin NX (TensorRT fp16)",
            "SOTA zero-shot on 2D-3D-S vs OmniStereo / VGGT / LightStereo",
        ],
        "datasets": datasets_card(),
        "benchmarks": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "fastvidar", **evaluation_demo_run()}
