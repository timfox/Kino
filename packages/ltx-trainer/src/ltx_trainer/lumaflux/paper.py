"""LumaFlux paper stub: framework card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lumaflux.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    lf = b["table1_benchmarks"]["hdrtv1k"]["lumaflux"]
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "authors": "Saini, Gedik, Birkbeck, Wang, Adsumilli, Bovik (UT Austin / Google)",
        "problem": (
            "8-bit BT.709 SDR must lift to 10-bit PQ BT.2020 HDR without clipped highlights, "
            "desaturated color, or unstable tone under real camera pipelines and compression."
        ),
        "method": {
            "backbone": "Frozen Flux MM-DiT; prompt-free flow-matching ODE (40 steps)",
            "pga": "Physically-guided LoRA on V with luminance, gradient, saturation, FFT gating",
            "pcm": "SigLIP FiLM cross-modulation for chroma/texture stability",
            "coupler": "HDR residual fusion of physical + perceptual streams (λ_t,ℓ)",
            "rqs": "Monotone rational-quadratic spline on VAE-decoded luma",
        },
        "results": {
            "hdrtv1k_psnr_db": lf["psnr"],
            "hdrtv1k_delta_e_itp": lf["delta_e_itp"],
            "luma_eval_psnr_db": b["table1_benchmarks"]["luma_eval"]["lumaflux"]["psnr"],
            "training_pairs": 318_000,
        },
        "reference_metrics": b,
        "integration": "Complements DiffHDR, LumiVid, stem2; stub only (no Flux weights).",
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.lumaflux.mock import evaluation_smoke

    return evaluation_smoke()
