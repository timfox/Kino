"""WaveNeXt 2 toy GAN/diffusion synthesis for smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.wavenext2.diffusion import sequential_submodel_inference
from ltx_trainer.wavenext2.gan import fixed_point_iteration


def toy_gan_synthesize(
    mel: np.ndarray,
    iterations: int = 4,
    *,
    seed: int = 0,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    y = fixed_point_iteration(mel, iterations, rng)
    rms = float(np.sqrt(np.mean(y**2)))
    return {"rms": rms, "waveform_len": len(y)}


def toy_diff_synthesize(
    mel: np.ndarray,
    noise_schedule: tuple[float, ...] | list[float],
    *,
    seed: int = 0,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    mel_len = len(np.asarray(mel).ravel())
    y0 = rng.standard_normal(mel_len * 300) * 0.05
    sched = tuple(noise_schedule)
    y = sequential_submodel_inference(mel, y0, sched, rng)
    rms = float(np.sqrt(np.mean(y**2)))
    return {"rms": rms, "waveform_len": len(y)}


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mel = rng.standard_normal(80)
    gan = toy_gan_synthesize(mel, iterations=3, seed=seed)
    diff = toy_diff_synthesize(mel, noise_schedule=(1.0, 0.5, 0.1), seed=seed)
    return {
        "gan_rms": round(gan["rms"], 4),
        "diff_rms": round(diff["rms"], 4),
        "gan_waveform_len": int(gan["waveform_len"]),
    }
