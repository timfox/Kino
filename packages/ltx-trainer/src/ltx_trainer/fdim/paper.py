"""FDIM paper stub: framework card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fdim.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle


def framework_card() -> dict[str, Any]:
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "authors": "Wang, Zhang, Zhuang, Zhang, Yu, Zhao (ZJU / Huawei)",
        "problem": (
            "NVCs introduce content-varying generative artifacts that break traditional VQA. "
            "Metrics must generalize across conventional and neural codecs, SDR and HDR."
        ),
        "method": {
            "deep": "ResNet-18 multi-scale → CAFM (deformable, ref-conditioned) → MSF (CBAM) → MLP",
            "trad": "VMAF proxy (DLM/VIF/TI-style hand-crafted fusion)",
            "fusion": "Per-component 4-parameter logistic (Eq. 1), average (Eq. 2)",
            "hdr": "PU21 preprocessing for zero-shot SDR→HDR (FDIM+PU21)",
            "training": "DCVQA 16k+ sequences, pairwise ranking loss (Eq. 19–21)",
        },
        "reference_metrics": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.fdim.mock import evaluation_smoke

    return evaluation_smoke()
