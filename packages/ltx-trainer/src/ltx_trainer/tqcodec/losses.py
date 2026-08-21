"""Loss functions and objective metrics (Sec. 3.1.3, 4.1)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.tqcodec.config import LOSS_WEIGHTS


def waveform_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target)


def mel_loss_stub(pred: Tensor, target: Tensor) -> Tensor:
    """Multi-scale mel L1 stub (Sec. 3.1.3)."""
    return F.l1_loss(pred, target) + F.l1_loss(torch.log1p(pred.abs()), torch.log1p(target.abs()))


def codec_generator_loss(
    pred: Tensor,
    target: Tensor,
    *,
    adv_loss: Tensor | None = None,
    feat_loss: Tensor | None = None,
    vq_commit: Tensor | None = None,
    vq_codebook: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    w = LOSS_WEIGHTS
    mel = mel_loss_stub(pred, target)
    wav = waveform_loss(pred, target)
    total = w["mel_multi_scale"] * mel + w["waveform"] * wav
    stats = {"mel": float(mel.detach()), "waveform": float(wav.detach())}
    if feat_loss is not None:
        total = total + w["feature_matching"] * feat_loss
        stats["feature"] = float(feat_loss.detach())
    if adv_loss is not None:
        total = total + w["adversarial"] * adv_loss
        stats["adversarial"] = float(adv_loss.detach())
    if vq_codebook is not None:
        total = total + w["codebook"] * vq_codebook
        stats["codebook"] = float(vq_codebook.detach())
    if vq_commit is not None:
        total = total + w["commitment"] * vq_commit
        stats["commitment"] = float(vq_commit.detach())
    return total, stats


def _stft_log_mag(x: Tensor, n_fft: int = 1024, hop: int = 256) -> Tensor:
    if x.dim() == 3:
        x = x.squeeze(1)
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    return torch.log1p(spec.abs().pow(2))


def log_spectral_distance(pred: Tensor, target: Tensor) -> float:
    """LSD metric (Sec. 4.1)."""
    x = _stft_log_mag(pred)
    y = _stft_log_mag(target)
    lsd_frames = torch.sqrt(((x - y) ** 2).mean(dim=1))
    return float(lsd_frames.mean())


def snr_db(pred: Tensor, target: Tensor, eps: float = 1e-8) -> float:
    noise = target - pred
    signal_power = target.pow(2).mean()
    noise_power = noise.pow(2).mean().clamp(min=eps)
    return float(10 * math.log10((signal_power / noise_power).clamp(min=eps)))
