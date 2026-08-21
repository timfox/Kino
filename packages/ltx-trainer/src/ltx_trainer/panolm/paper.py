"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panolm.benchmarks import benchmarks_bundle, table5_ours
from ltx_trainer.panolm.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    ref = table5_ours()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "paradigm": "Panorama-Language Modeling (PLM)",
        "task": "Panoramic visual question answering on adverse omni-scenes (PanoVQA)",
        "method": {
            "PSA": "Panoramic Sparse Attention with Top-K + position gate",
            "PHA": "Parallel SWA + PSA in ViT blocks",
            "base": "Qwen2.5-VL + full SFT on PanoVQA",
            "dataset": "653K QA — Normal / Occluded / Accident",
        },
        "results": {
            "PanoLM_7B_avg_gpt": ref["avg"],
            "1pano_vs_6cam_SFT": "41.42 vs 40.22 on PanoVQA-mini (3B)",
            "beats_Qwen32B_zero_shot": ref["avg"] > 35.56,
        },
        "reference_metrics": benchmarks_bundle(),
        "integration": (
            "GOPEX implements PSA/PHA stubs, PanoVQA taxonomy, and Table 5/6/8 anchors. "
            "Production: github.com/InSAI-Lab/PanoVQA."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.panolm.mock import evaluation_smoke

    return evaluation_smoke()
