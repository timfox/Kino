"""DSFA — proxy-to-wild CodecFake detection (arXiv:2606.07494)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DsfaConfig:
    paper_arxiv: str = "arXiv:2606.07494"
    title: str = "Mitigating Proxy-to-Wild Domain Gap in Deepfake Speech"
    framework: str = "DSFA"
    backbone: str = "PT-Wav2Vec2-Large-AntiDeepfake"
    backbone_hf: str = "nii-yamagishilab/xls-r-2b-anti-deepfake"

    # Training (§4)
    sample_rate_hz: int = 16000
    clip_s: float = 4.0
    batch_size: int = 14
    learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    supcon_lambda: float = 0.1

    # DSFA (§3.2)
    dsfa_prob: float = 0.25
    dsfa_layer_uniform: int = 24
    dsfa_layer_gaussian: int = 1
    noise_uniform: bool = True

    # CoSG datasets (Table 1)
    cosg_eval_spoof_models: int = 17
    cosg_exteval_spoof_models: int = 40
    cors_codecs: int = 31

    # Table 2 — model (k) DSFA-only
    cosg_eval_eer_k: float = 3.00
    cosg_exteval_eer_k: float = 21.80

    # Table 2 — model (j) DSFA + SupCon
    cosg_eval_eer_j: float = 2.78
    cosg_exteval_eer_j: float = 23.00

    # Table 2 — model (g) PT backbone baseline
    cosg_eval_eer_g: float = 3.95
    cosg_exteval_eer_g: float = 22.19

    # Table 2 — model (f) DEC balance baseline
    cosg_exteval_eer_f: float = 27.07

    # Table 4 — optimal p on ExtEval
    cosg_exteval_eer_p025: float = 22.77

    # Fig 2 overlap gains
    mean_overlap_baseline: float = 42.91
    mean_overlap_dsfa: float = 43.03
    std_overlap_baseline: float = 65.01
    std_overlap_dsfa: float = 67.09
