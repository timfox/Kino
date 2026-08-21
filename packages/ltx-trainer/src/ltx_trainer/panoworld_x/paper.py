"""Paper card export."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoworld_x.benchmarks import benchmarks_bundle
from ltx_trainer.panoworld_x.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL
from ltx_trainer.panoworld_x.datasets import datasets_card
from ltx_trainer.panoworld_x.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "authors": [
            "Yuyang Yin",
            "HaoXiang Guo",
            "Fangfu Liu",
            "Mengyu Wang",
            "Hanwen Liang",
            "Eric Li",
            "Yikai Wang",
            "Xiaojie Jin",
            "Yao Zhao",
            "Yunchao Wei",
        ],
        "affiliation": "BJTU / Skywork AI / Tsinghua / UofT / BNU",
        "components": [
            "PanoExplorer: UE 504-scene routes + 116k panoramic videos",
            "Exploration-Aware Attention (Plücker route + ControlNet-style branch)",
            "Sphere-Aware Attention (Haversine mask, Eq. 6–7)",
            "Explorable Sphere-Aware DiT on CogVideoX-5B-I2V",
        ],
        "contributions": [
            "Large-scale explorable panoramic video dataset with 6-DoF routes",
            "Sphere-aware latent attention for ERP seam continuity",
            "SOTA PSNR/SSIM/FID/FVD vs 360DVD, Imagine360, GenEX",
            "Precise camera control vs CameraCtrl / AC3D on perspective crops",
        ],
        "datasets": datasets_card(),
        "benchmarks": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "panoworld_x", **evaluation_demo_run()}
