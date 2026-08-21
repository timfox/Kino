"""Video-aligned LTX audio VAE encode (no ltx_trainer dependency).

Used by inference pipelines and optionally extended in ``ltx_trainer.audio_encode``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from ltx_core.model.audio_vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class AudioAlignConfig:
    align_to_video: bool = True
    pad_latent_tail: bool = True


def align_config_from_env() -> AudioAlignConfig:
    return AudioAlignConfig(
        align_to_video=_env_bool("GOPEX_AUDIO_ALIGN_VIDEO", True),
        pad_latent_tail=_env_bool("GOPEX_AUDIO_PAD_LATENT", True),
    )


def latent_frames_per_second(
    *,
    sample_rate: int,
    mel_hop_length: int,
    downsample: int = LATENT_DOWNSAMPLE_FACTOR,
) -> float:
    return float(sample_rate) / float(mel_hop_length) / float(downsample)


def required_latent_frames_for_video(
    *,
    video_pixel_frames: int,
    video_fps: float,
    sample_rate: int,
    mel_hop_length: int,
) -> int:
    duration = float(video_pixel_frames) / max(float(video_fps), 1e-6)
    return AudioLatentShape.from_duration(
        batch=1,
        duration=duration,
        sample_rate=sample_rate,
        hop_length=mel_hop_length,
        audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
    ).frames


def prepare_waveform_duration(
    audio: Audio,
    *,
    video_pixel_frames: int,
    video_fps: float,
) -> tuple[Audio, dict[str, Any]]:
    """Trim or zero-pad waveform to the video clip duration (wall-clock seconds)."""
    meta: dict[str, Any] = {}
    wf = audio.waveform
    if wf.dim() == 2:
        wf = wf.unsqueeze(0)
    duration_sec = float(video_pixel_frames) / max(float(video_fps), 1e-6)
    target_samples = int(round(duration_sec * float(audio.sampling_rate)))
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
    meta["duration_sec"] = round(duration_sec, 4)
    return Audio(waveform=wf, sampling_rate=audio.sampling_rate), meta


def conform_audio_latent_time(
    latents: torch.Tensor,
    expected_time_steps: int,
    *,
    pad_tail: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Align ``[C, T, F]`` or ``[B, C, T, F]`` along the time dimension."""
    info: dict[str, Any] = {"expected_time_steps": expected_time_steps}
    if latents.dim() == 3:
        current = int(latents.shape[1])
    elif latents.dim() == 4:
        current = int(latents.shape[2])
    else:
        info["skipped"] = "rank_not_supported"
        return latents, info

    info["time_steps_before"] = current
    if current == expected_time_steps:
        info["time_steps_after"] = current
        return latents, info

    if current > expected_time_steps:
        if latents.dim() == 3:
            out = latents[:, :expected_time_steps, :]
        else:
            out = latents[:, :, :expected_time_steps, :]
        info["align_op"] = "trim"
    elif pad_tail:
        pad = expected_time_steps - current
        if latents.dim() == 3:
            pad_tensor = torch.zeros(
                latents.shape[0],
                pad,
                latents.shape[2],
                device=latents.device,
                dtype=latents.dtype,
            )
            out = torch.cat([latents, pad_tensor], dim=1)
        else:
            pad_tensor = torch.zeros(
                latents.shape[0],
                latents.shape[1],
                pad,
                latents.shape[3],
                device=latents.device,
                dtype=latents.dtype,
            )
            out = torch.cat([latents, pad_tensor], dim=2)
        info["align_op"] = "pad"
    else:
        out = latents
        info["align_op"] = "no_pad_short"

    info["time_steps_after"] = int(out.shape[1] if out.dim() == 3 else out.shape[2])
    return out, info


def encode_audio_tensor_for_inference(
    audio: Audio,
    audio_encoder: torch.nn.Module,
    audio_processor: AudioProcessor | None,
    output_shape: VideoPixelShape,
    *,
    cfg: AudioAlignConfig | None = None,
) -> torch.Tensor:
    """Encode waveform → ``[1, C, T, F]`` with *T* matched to ``output_shape``."""
    cfg = cfg or align_config_from_env()
    prepared = audio
    if cfg.align_to_video:
        prepared, _prep = prepare_waveform_duration(
            audio,
            video_pixel_frames=int(output_shape.frames),
            video_fps=float(output_shape.fps),
        )

    device = next(audio_encoder.parameters()).device
    dtype = next(audio_encoder.parameters()).dtype
    if audio_processor is None:
        audio_processor = AudioProcessor(
            target_sample_rate=audio_encoder.sample_rate,
            mel_bins=audio_encoder.mel_bins,
            mel_hop_length=audio_encoder.mel_hop_length,
            n_fft=audio_encoder.n_fft,
        ).to(device=device)

    waveform = prepared.waveform.to(device=device, dtype=dtype)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    mel_spectrogram = audio_processor.waveform_to_mel(
        Audio(waveform=waveform, sampling_rate=prepared.sampling_rate)
    )
    latents = audio_encoder(mel_spectrogram.to(dtype=dtype)).squeeze(0)

    if cfg.align_to_video:
        need_t = required_latent_frames_for_video(
            video_pixel_frames=int(output_shape.frames),
            video_fps=float(output_shape.fps),
            sample_rate=int(audio_encoder.sample_rate),
            mel_hop_length=int(audio_encoder.mel_hop_length),
        )
        latents, _ = conform_audio_latent_time(latents, need_t, pad_tail=cfg.pad_latent_tail)

    if latents.dim() == 3:
        latents = latents.unsqueeze(0)
    return latents.to(device=device, dtype=dtype)
