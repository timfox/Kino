"""IRAF — interference-resilient adaptive fusion (Zhong et al., arXiv:2606.06559)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IrafConfig:
    paper_arxiv: str = "arXiv:2606.06559"
    title: str = (
        "IRAF: Interference-Resilient Adaptive Fusion for Noise-Robust "
        "End-to-End Full-Duplex Spoken Dialogue Systems"
    )
    framework: str = "IRAF"

    # §2.1 duplex backbone
    speech_encoder: str = "FastConformer streaming 100M (12.5 Hz, 80 ms right context)"
    llm_backbone: str = "TinyLlama-1.1B"
    speech_codec: str = "NanoCodec 0.6 kbps (4×4037 @ 12.5 Hz)"
    speech_decoder_layers: int = 12
    frame_rate_hz: float = 12.5
    inter_turn_pause_s: float = 0.64
    barge_in_prob: float = 0.5

    # §2.2 IRAF module
    speaker_encoder: str = "ECAPA-TDNN"
    iraf_transformer_layers: int = 1
    gate_scale: float = 2.0  # g_t = 2 * sigmoid(...)

    # Loss weights (Eq. 1 + §2.2 auxiliary)
    lambda_text: float = 1.0
    lambda_audio: float = 5.0
    lambda_gate: float = 0.1

    # Training (§4.1)
    peak_lr: float = 3e-4
    warmup_steps: int = 2500
    grad_clip: float = 1.0
    snr_db_ms_marco: tuple[float, float] = (0.0, 10.0)
    snr_db_instructs2s: tuple[float, float] = (0.0, 20.0)

    # Table 1 — MS MARCO, MUSAN speech interference, ALL
    msmarco_noisyaug_bleu: float = 12.74
    msmarco_noisyaug_sbert: float = 0.506
    msmarco_noisyaug_rsr_pct: float = 93.1
    msmarco_iraf_bleu: float = 14.20
    msmarco_iraf_sbert: float = 0.523
    msmarco_iraf_rl_s: float = 0.96
    msmarco_iraf_rsr_pct: float = 95.7
    msmarco_iraf_bleu_rel_pct: float = 11.46

    # Table 2 — InstructS2S-200K, interfering speakers only
    instruct_noisyaug_bleu: float = 9.64
    instruct_noisyaug_rsr_pct: float = 69.2
    instruct_iraf_bleu: float = 13.76
    instruct_iraf_sbert: float = 0.58
    instruct_iraf_rl_s: float = 0.82
    instruct_iraf_rsr_pct: float = 91.0
    instruct_iraf_sl_s: float = 0.73
    instruct_iraf_ssr_pct: float = 99.8
    instruct_iraf_bleu_rel_pct: float = 42.73

    datasets: tuple[str, ...] = ("MS MARCO (CosyVoice2 TTS)", "InstructS2S-200K")
    noise_corpus: str = "MUSAN (speech interference + background noise)"
