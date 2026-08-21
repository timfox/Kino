"""TQCodec configuration (He et al., arXiv:2603.01592)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = "https://arxiv.org/abs/2603.01592"
PAPER_DOI = "10.48550/arXiv.2603.01592"
PAPER_TITLE = "TQCodec: Towards Neural Audio Codec for High-Fidelity Music Streaming"

SAMPLE_RATE = 44100
BITRATES_KBPS = (32, 64, 128)
CODEBOOKS_BY_BITRATE = {32: 5, 64: 5, 128: 10}  # stereo configs per paper Sec. 4.1
CODEBOOK_SIZE = 512
FRAME_RATE = 689

# SEANet decoder budget (Sec. 3.1.1)
DECODER_GMACS = 6.31
ENCODER_GMACS_DEFAULT = 2.0
ENCODER_GMACS_IMBALANCED = 80.0
RECEPTIVE_FIELD_SAMPLES = 2410
DAC_DECODER_GMACS = 365.0
DAC_RECEPTIVE_FIELD = 17706

# Architecture (Sec. 4.1)
DOWNSAMPLE_FACTORS = (2, 4, 8)
ENCODER_BLOCKS = 3
DECODER_BLOCKS = 3
ENCODER_DIM = 64
LATENT_DIM = 128
DECODER_DIM = 128

# PQMF subband (Sec. 3.2)
PQMF_SUBBANDS = 16
CORE_SUBBANDS = 12
CORE_LATENT_DIM = 128
HIGH_SUBBAND_LATENT_DIM = 6

# Loss weights (Sec. 3.1.3)
LOSS_WEIGHTS = {
    "mel_multi_scale": 15.0,
    "waveform": 1.0,
    "feature_matching": 2.0,
    "adversarial": 1.0,
    "codebook": 1.0,
    "commitment": 0.25,
}

# Training (Sec. 4.1)
TRAIN_ITERATIONS = 400_000
BATCH_SIZE = 32
CLIP_SECONDS = 1.0
LEARNING_RATE = 1e-4
ADAM_B1 = 0.8
ADAM_B2 = 0.9


@dataclass
class TQCodecConfig:
    sample_rate: int = SAMPLE_RATE
    bitrate_kbps: int = 64
    num_codebooks: int = 5
    codebook_size: int = CODEBOOK_SIZE
    encoder_dim: int = ENCODER_DIM
    latent_dim: int = LATENT_DIM
    decoder_dim: int = DECODER_DIM
    use_rsimvq: bool = True
    use_waveform_loss: bool = True
    use_subband: bool = True
    imbalanced_encoder: bool = True

    @property
    def frame_rate(self) -> float:
        hop = 1
        for f in DOWNSAMPLE_FACTORS:
            hop *= f
        return self.sample_rate / hop
