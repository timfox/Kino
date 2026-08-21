"""AG-REPA configuration (arXiv:2603.01006)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgRepaConfig:
    """Attribution-Guided REPA for token-conditioned audio Flow Matching."""

    paper_arxiv: str = "2603.01006"
    github_url: str = "https://github.com/zpforlove/AG-REPA"
    num_dit_layers: int = 24
    top_k_layers: int = 3
    warmup_probe_steps: int = 5_000
    train_checkpoint_steps: int = 500_000
    fog_probe_interval: int = 200
    fog_probe_batch_size: int = 2
    semantic_teacher: str = "Whisper-large-v3"
    acoustic_teacher: str = "BEATs"
    speech_dataset: str = "LibriSpeech"
    audio_dataset: str = "AudioSet"
    token_config_a: str = "S3 + AudioSet"
    token_config_b: str = "Config A + interleaved BEATs (1:1)"
    lambda_bit: float = 1.0
    lambda_repa_base: float = 1.0
    fog_epsilon: float = 1e-6
    convergence_fad_target: float = 1.5
    random_seed: int = 42
    paper_top3_speech_fog: tuple[int, ...] = (1, 2, 7)
    paper_top3_audio_fog: tuple[int, ...] = (1, 7, 2)

    @classmethod
    def production(cls) -> AgRepaConfig:
        return cls()
