"""DiffHDR paper stub: framework card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.diffhdr.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    t1 = b["table1_si_hdr"]["ours"]
    t4 = b["table4_log_gamma_vae"]["ours"]
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "authors": "Yu, Ma, He, Isikdogan, Xu, Smirnov, Salamanca, Mi, Delgado, Yu, Philip, Li, Wang, Debevec (TAMU / Eyeline / Netflix)",
        "problem": (
            "8-bit LDR video loses highlight/shadow radiance; feed-forward ITM cannot plausibly "
            "inpaint clipped regions. Single-LDR→HDR is one-to-many without generative priors."
        ),
        "method": {
            "representation": "Log-Gamma maps HDR into pretrained Wan VAE domain (no VAE finetune)",
            "backbone": "VACE video-to-video DiT + rank-32 LoRA, rectified flow matching",
            "data": "~5400×81-frame clips from Polyhaven HDRIs (Blender skybox renders)",
            "control": "Luminance masks (τ_high=0.95, τ_low=0.05, EMA α=0.7), context-focused cross-attention, ref image",
        },
        "results": {
            "si_hdr_pu21_piqe": t1["pu21_piqe"],
            "si_hdr_fid": t1["fid"],
            "log_gamma_vae_psnr_db": t4["psnr"],
            "polyhaven_fovvideovdp": b["table2_video"]["ours_polyhaven"]["fovvideovdp"],
        },
        "reference_metrics": b,
        "integration": "Complements LumiVid LogC3 path and sdr2hdr pipeline; stub only (no Wan-14B weights).",
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.diffhdr.mock import evaluation_smoke

    return evaluation_smoke()
