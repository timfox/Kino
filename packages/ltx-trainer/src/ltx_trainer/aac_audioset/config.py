"""AAC + AudioSet semantics — Gupta et al., arXiv:2606.05717."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AacAudiosetConfig:
    paper_arxiv: str = "arXiv:2606.05717"
    title: str = "Enhancing Audio Captioning with Auxiliary AudioSet Semantics"
    framework: str = "AAC-AudioSet"

    # Datasets (§3.1)
    clotho_samples: int = 4981
    audiocaps_train_clips: int = 50000
    audiocaps_eval_captions_per_clip: int = 5

    # Architecture (§2, §3.2)
    sample_rate_hz: int = 32000
    mel_bins: int = 224
    mel_window_ms: float = 32.0
    mel_hop_ms: float = 10.0
    audioset_class_count: int = 527
    top_k_keywords: int = 5
    model_dim: int = 768
    attention_heads: int = 8
    ffn_hidden: int = 2048
    encoder_layers: int = 3
    decoder_layers: int = 3
    params_m: float = 110.0
    flops_g: float = 88.55
    beam_width: int = 5

    # Table 1 — Clotho in-domain (train Clotho, test Clotho)
    clotho_id_spider: float = 0.286
    clotho_id_fense: float = 0.478
    clotho_id_bleu1: float = 0.602
    clotho_id_cider: float = 0.446

    # Table 1 — Clotho cross (train AudioCaps, test Clotho)
    clotho_xd_spider: float = 0.142
    clotho_xd_fense: float = 0.437

    # Table 2 — AudioCaps in-domain (train AudioCaps, test AudioCaps)
    audiocaps_id_spider: float = 0.470
    audiocaps_id_fense: float = 0.615
    audiocaps_id_bleu1: float = 0.716
    audiocaps_id_cider: float = 0.780

    # Table 2 — AudioCaps cross (train Clotho, test AudioCaps)
    audiocaps_xd_spider: float = 0.214
    audiocaps_xd_fense: float = 0.419

    # Table 1/2 — Kim et al. Clotho in-domain baseline
    kim_clotho_id_spider: float = 0.255

    # Table 1/2 — Pengi (large pretrained) Clotho / AudioCaps
    pengi_clotho_spider: float = 0.260
    pengi_audiocaps_spider: float = 0.256

    # Table 3 — keyword ablation (Ours, Clotho)
    ours_no_kw_spider: float = 0.260
    ours_kw_spider: float = 0.286
    ours_no_kw_bleu1: float = 0.588
    ours_kw_bleu1: float = 0.602

    # Table 4 — K sensitivity (Clotho)
    k5_spider: float = 0.286
    k10_spider: float = 0.282
    k15_spider: float = 0.282

    # Keyword→LLaMA-2-7B baseline (§3.3)
    llama_clotho_spider: float = 0.095
    llama_audiocaps_spider: float = 0.103
