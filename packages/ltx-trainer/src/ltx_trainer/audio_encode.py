"""Improved LTX audio VAE encode path (SwanSphere / Foley-Omni / OmniAudio ideas).

Papers integrated as **preprocess + metadata** (no replacement weights):

- **SwanSphere** (arXiv:2605.30940): continuous mel latents, video-aligned frame budget,
  FOA W/X/Y/Z downmix proxy for stereo mel VAE, spatial azimuth sidecar.
- **Foley-Omni** (arXiv:2606.03672): duration locked to video, sync frame rate metadata for V2ST.
- **OmniAudio / Sphere360**: ``has_foa_audio`` and ERP spatial flags on shards.

Env toggles (default ON where noted):

- ``GOPEX_AUDIO_ALIGN_VIDEO=1`` — trim/pad waveform + latent *T* to video duration.
- ``GOPEX_AUDIO_FOA_DOWNMIX=1`` — 4-ch FOA → stereo proxy before mel (SwanSphere Appendix B.3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from ltx_core.model.audio_vae import AudioProcessor
from ltx_core.model.audio_vae.align_encode import (
    AudioAlignConfig,
    conform_audio_latent_time,
    encode_audio_tensor_for_inference as _core_encode_audio_tensor_for_inference,
    latent_frames_per_second,
    required_latent_frames_for_video,
)
from ltx_core.model.audio_vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR
from ltx_core.types import Audio

try:
    from ltx_trainer.swansphere.foa import FOA_CHANNEL_NAMES, intensity_vector_azimuth
except ImportError:  # pragma: no cover
    FOA_CHANNEL_NAMES = ("W", "X", "Y", "Z")
    intensity_vector_azimuth = None  # type: ignore[misc, assignment]


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class AudioEncodeConfig:
    align_to_video: bool = True
    foa_downmix: bool = True
    pad_latent_tail: bool = True
    # SwanSphere continuous-latent reference rate (metadata only for LTX mel VAE).
    reference_latent_fps: float = 21.5


def config_from_env() -> AudioEncodeConfig:
    return AudioEncodeConfig(
        align_to_video=_env_bool("GOPEX_AUDIO_ALIGN_VIDEO", True),
        foa_downmix=_env_bool("GOPEX_AUDIO_FOA_DOWNMIX", True),
        pad_latent_tail=_env_bool("GOPEX_AUDIO_PAD_LATENT", True),
        reference_latent_fps=float(os.environ.get("GOPEX_AUDIO_REFERENCE_LATENT_FPS", "21.5")),
    )


def _waveform_channel_layout(waveform: torch.Tensor) -> tuple[str, int]:
    """Return layout tag and channel count (batch stripped)."""
    if waveform.dim() == 3:
        ch = int(waveform.shape[1])
    elif waveform.dim() == 2:
        ch = int(waveform.shape[0])
    else:
        ch = 1
    if ch == 4:
        return "foa_wxyz", ch
    if ch == 2:
        return "stereo", ch
    if ch == 1:
        return "mono", ch
    return f"multi_{ch}", ch


def foa_to_stereo_proxy(
    waveform: torch.Tensor,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """
    SwanSphere curriculum downmix: W = L+R proxy, one axis carries L−R (Appendix B.3).
    Input: ``[4, samples]`` or ``[1, 4, samples]``.
    """
    if waveform.dim() == 3:
        w = waveform[0]
    else:
        w = waveform
    if w.shape[0] != 4:
        raise ValueError(f"foa_to_stereo_proxy expects 4 channels, got {w.shape[0]}")
    w_ch, x_ch, y_ch, z_ch = w[0], w[1], w[2], w[3]
    left = w_ch + x_ch
    right = w_ch - x_ch
    stereo = torch.stack([left, right], dim=0)
    return stereo.unsqueeze(0) if waveform.dim() == 3 else stereo


def prepare_waveform_for_vae(
    audio: Audio,
    *,
    cfg: AudioEncodeConfig | None = None,
    video_pixel_frames: int | None = None,
    video_fps: float | None = None,
) -> tuple[Audio, dict[str, Any]]:
    """Resample (later), mono→stereo, optional FOA downmix, optional duration trim."""
    cfg = cfg or config_from_env()
    meta: dict[str, Any] = {}
    wf = audio.waveform
    if wf.dim() == 2:
        wf = wf.unsqueeze(0)
    layout, ch = _waveform_channel_layout(wf)
    meta["input_layout"] = layout
    meta["input_channels"] = ch

    if layout == "foa_wxyz" and intensity_vector_azimuth is not None:
        try:
            raw4 = wf[0] if wf.dim() == 3 else wf
            if raw4.shape[0] == 4:
                az = intensity_vector_azimuth(
                    raw4[0].detach().cpu().numpy(),
                    raw4[1].detach().cpu().numpy(),
                    raw4[2].detach().cpu().numpy(),
                )
                meta["foa_azimuth_rad"] = round(float(az), 4)
        except Exception:
            pass

    if layout == "foa_wxyz" and cfg.foa_downmix:
        wf = foa_to_stereo_proxy(wf)
        meta["foa_downmix"] = "stereo_proxy_wx"
        layout = "stereo"
    elif layout == "mono" and wf.shape[1] == 1:
        wf = wf.repeat(1, 2, 1)
        meta["mono_upmix"] = "duplicate"

    if cfg.align_to_video and video_pixel_frames is not None and video_fps is not None:
        target_samples = int(round(float(video_pixel_frames) / max(float(video_fps), 1e-6) * audio.sampling_rate))
        current = int(wf.shape[-1])
        if target_samples > 0 and current != target_samples:
            if current > target_samples:
                wf = wf[..., :target_samples]
                meta["duration_trim"] = "head"
            else:
                pad = target_samples - current
                wf = torch.nn.functional.pad(wf, (0, pad))
                meta["duration_trim"] = "pad"
            meta["target_samples"] = target_samples

    return Audio(waveform=wf, sampling_rate=audio.sampling_rate), meta


def build_timing_meta(
    *,
    encoder: torch.nn.Module,
    duration_sec: float,
    video_fps: float | None,
    video_pixel_frames: int | None,
    cfg: AudioEncodeConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or config_from_env()
    sr = int(getattr(encoder, "sample_rate", 16000))
    hop = int(getattr(encoder, "mel_hop_length", 160))
    lfps = latent_frames_per_second(sample_rate=sr, mel_hop_length=hop)
    meta = {
        "latent_fps": round(lfps, 4),
        "mel_hop_length": hop,
        "vae_sample_rate": sr,
        "latent_downsample": LATENT_DOWNSAMPLE_FACTOR,
        "duration_sec": round(float(duration_sec), 4),
        "continuous_latent": True,
        "reference_latent_fps_swan": cfg.reference_latent_fps,
    }
    if video_fps is not None and video_pixel_frames is not None:
        meta["video_fps"] = float(video_fps)
        meta["video_pixel_frames"] = int(video_pixel_frames)
        meta["required_latent_frames"] = required_latent_frames_for_video(
            video_pixel_frames=video_pixel_frames,
            video_fps=video_fps,
            sample_rate=sr,
            mel_hop_length=hop,
        )
        meta["sync_offset_frames"] = int(meta["required_latent_frames"]) - int(
            round(duration_sec * lfps)
        )
    return meta


def encode_audio_bundle(
    audio_vae_encoder: torch.nn.Module,
    audio_processor: AudioProcessor,
    audio: Audio,
    *,
    cfg: AudioEncodeConfig | None = None,
    video_pixel_frames: int | None = None,
    video_fps: float | None = None,
) -> dict[str, torch.Tensor | int | float | dict[str, Any]]:
    """
    Encode waveform → LTX audio latents with paper-inspired preprocessing and metadata.

    Returns the same keys as legacy ``encode_audio`` in ``process_videos.py``, plus
    ``audio_encode`` timing/sync block for folds and audits.
    """
    cfg = cfg or config_from_env()
    prepared, prep_meta = prepare_waveform_for_vae(
        audio,
        cfg=cfg,
        video_pixel_frames=video_pixel_frames,
        video_fps=video_fps,
    )

    device = next(audio_vae_encoder.parameters()).device
    dtype = next(audio_vae_encoder.parameters()).dtype
    waveform = prepared.waveform.to(device=device, dtype=dtype)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    duration = float(waveform.shape[-1]) / float(prepared.sampling_rate)
    mel_spectrogram = audio_processor.waveform_to_mel(
        Audio(waveform=waveform, sampling_rate=prepared.sampling_rate)
    )
    mel_spectrogram = mel_spectrogram.to(dtype=dtype)
    latents = audio_vae_encoder(mel_spectrogram)
    latents = latents.squeeze(0)

    align_info: dict[str, Any] = {}
    if cfg.align_to_video and video_pixel_frames is not None and video_fps is not None:
        need_t = required_latent_frames_for_video(
            video_pixel_frames=video_pixel_frames,
            video_fps=video_fps,
            sample_rate=int(audio_vae_encoder.sample_rate),
            mel_hop_length=int(audio_vae_encoder.mel_hop_length),
        )
        latents, align_info = conform_audio_latent_time(
            latents, need_t, pad_tail=cfg.pad_latent_tail
        )

    _c, time_steps, freq_bins = latents.shape

    timing = build_timing_meta(
        encoder=audio_vae_encoder,
        duration_sec=duration,
        video_fps=video_fps,
        video_pixel_frames=video_pixel_frames,
        cfg=cfg,
    )
    timing["prep"] = prep_meta
    timing["align"] = align_info

    return {
        "latents": latents,
        "num_time_steps": int(time_steps),
        "frequency_bins": int(freq_bins),
        "duration": duration,
        "audio_encode": timing,
    }


def encode_audio_tensor_for_inference(
    audio: Audio,
    audio_encoder: torch.nn.Module,
    audio_processor: AudioProcessor | None,
    output_shape: Any,
) -> torch.Tensor:
    """Pipeline helper: returns ``[1, C, T, F]`` aligned to ``VideoPixelShape``."""
    cfg = config_from_env()
    align_cfg = AudioAlignConfig(
        align_to_video=cfg.align_to_video,
        pad_latent_tail=cfg.pad_latent_tail,
    )
    return _core_encode_audio_tensor_for_inference(
        audio,
        audio_encoder,
        audio_processor,
        output_shape,
        cfg=align_cfg,
    )
