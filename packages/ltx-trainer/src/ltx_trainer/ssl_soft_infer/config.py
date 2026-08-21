"""Soft SSL discrete-token inference (Onda et al., arXiv:2606.06806)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SslModel = Literal["hubert_large", "wavlm_large"]
AssignmentMode = Literal["hard", "soft"]


@dataclass
class SslSoftInferConfig:
    paper_arxiv: str = "arXiv:2606.06806"
    title: str = (
        "Leveraging Soft Distributions of SSL-Derived Discrete Speech Tokens "
        "for Downstream Inference"
    )
    framework: str = "SSL-Soft-Infer"
    demo_url: str = "https://ondatk68.github.io/onda-demo/projects/soft-token-inference/"

    hubert_hf: str = "facebook/hubert-large-ll60k"
    wavlm_hf: str = "microsoft/wavlm-large"
    ssl_layer: int = 21
    kmeans_hours: float = 30.0
    cluster_sizes: tuple[int, ...] = (128, 1024, 4096)

    train_data_asr: str = "LibriSpeech-100h"
    train_data_synth: str = "LJSpeech"
    asr_train_assign: AssignmentMode = "hard"
    asr_infer_assign: AssignmentMode = "soft"

    tau_librispeech: float = 8.0
    tau_ted2: float = 8.0
    tau_chime4: float = 8.0
    tau_erj: float = 13.5
    tau_synth: float = 8.0

    # Table 1 — WavLM K=4096 hard/soft ERJ beats continuous (38.8 vs 38.9 cont.)
    wavlm4096_erj_soft: float = 38.8
    wavlm_cont_erj: float = 38.9

    # Table 2 — WavLM K=4096 out-of-domain VC SpkSim
    wavlm4096_vc_spksim_soft: float = 0.820
    wavlm4096_vc_spksim_hard: float = 0.806

    # Table 3 — WavLM 4096 ASR separability ratio hard→soft
    wavlm4096_asr_ratio_hard: float = 1.16
    wavlm4096_asr_ratio_soft: float = 1.22

    # Table 4 — multi-layer WavLM ERJ soft(ii)
    wavlm4layer_erj_soft_ii: float = 39.1

    downstream_tasks: tuple[str, ...] = ("asr", "speech_resynthesis", "voice_conversion")
    ood_datasets: tuple[str, ...] = ("TED-LIUM v2", "CHiME4", "ERJ")

    multi_layer_indices: tuple[int, ...] = (9, 15, 21, 22)
    multi_layer_tau_erj: tuple[float, ...] = (32.0, 32.0, 13.5, 21.0)
    multi_layer_tau_default: tuple[float, ...] = (4.0, 6.0, 8.0, 12.0)
