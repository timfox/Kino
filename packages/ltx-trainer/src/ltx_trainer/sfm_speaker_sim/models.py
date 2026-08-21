"""Representative model profiles and taxonomy (Sec. 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.sfm_speaker_sim.config import SfmSpeakerSimConfig


@dataclass(frozen=True)
class ModelProfile:
    name: str
    family: str
    layer_max_lcc: float
    layer_slope_lcc: float
    is_decoder: bool = False
    is_multilingual: bool = False
    is_ssl: bool = False

    def layerwise_lcc(self, n_layers: int = 13) -> list[float]:
        """Synthetic layer-wise LCC curve from max + slope (normalized layer index)."""
        xs = [i / max(n_layers - 1, 1) for i in range(n_layers)]
        base = self.layer_max_lcc - self.layer_slope_lcc * 0.5
        return [float(base + self.layer_slope_lcc * x) for x in xs]


def representative_models(cfg: SfmSpeakerSimConfig | None = None) -> list[ModelProfile]:
    c = cfg or SfmSpeakerSimConfig()
    return [
        ModelProfile("wavlm-large", "speech_ssl", c.wavlm_large_layer_max_lcc, -0.08, is_ssl=True),
        ModelProfile("wavlm-ssl_sv", "speech_ssl", 0.34, -0.02, is_ssl=True),
        ModelProfile("wavlm-base+", "speech_ssl", 0.31, -0.10, is_ssl=True),
        ModelProfile("whisper-large", "supervised_asr", c.whisper_large_layer_max_lcc, -0.06),
        ModelProfile("parakeet-tdt-1.1b", "supervised_asr", 0.28, -0.05),
        ModelProfile("qwen3-tts-12hz-0.6b-base", "supervised_tts", c.qwen3_tts_layer_max_lcc, -0.04, is_decoder=True),
        ModelProfile("speecht5-tts (decoder)", "supervised_tts", 0.15, -0.03, is_decoder=True),
        ModelProfile("audiogen-medium", "supervised_tta", c.audiogen_layer_max_lcc, 0.12),
        ModelProfile("ast_audioset_10_10_0.4593", "supervised_audio_cls", 0.18, -0.02),
        ModelProfile("atstframe_base", "audio_ssl", 0.16, 0.00, is_ssl=True),
        ModelProfile("hubert-large-ll60k", "speech_ssl", 0.30, -0.04, is_ssl=True),
        ModelProfile("hubert-large-ls960-ft", "speech_ssl", 0.27, -0.14, is_ssl=False),
        ModelProfile("wav2vec2-large", "speech_ssl", 0.29, -0.05, is_ssl=True),
        ModelProfile("wav2vec2-large-960h", "speech_ssl", 0.24, -0.12, is_ssl=False),
    ]


def model_taxonomy(cfg: SfmSpeakerSimConfig | None = None) -> dict[str, Any]:
    c = cfg or SfmSpeakerSimConfig()
    profiles = representative_models(c)
    return {
        "n_models_evaluated": c.n_models,
        "families": list(c.model_families),
        "datasets": list(c.datasets),
        "representative": [
            {
                "name": p.name,
                "family": p.family,
                "layer_max_lcc": p.layer_max_lcc,
                "layer_slope_lcc": p.layer_slope_lcc,
                "is_decoder": p.is_decoder,
                "is_ssl": p.is_ssl,
            }
            for p in profiles
        ],
    }
