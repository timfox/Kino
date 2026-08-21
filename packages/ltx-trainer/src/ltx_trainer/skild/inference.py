"""SKILD inference: spectrum I/O, timestep selection, SR / generation loops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ltx_trainer.skild.config import SkildConfig, SkildScheduleConfig
from ltx_trainer.skild.dct import frequency_radius_grid
from ltx_trainer.skild.diffusion import (
    dct_to_image,
    forward_marginal_dct,
    image_to_dct,
    reverse_step_dct,
)
from ltx_trainer.skild.schedule import SkildSchedule
from ltx_trainer.skild.spectrum import (
    PowerLawSpectrumFit,
    fit_power_law_spectrum,
    load_images_from_dir,
    radial_variance_spectrum,
    spectrum_map_from_fit,
)


@dataclass
class SkildSpectrum:
    """Per-mode variance ``S0`` and optional radial fit metadata."""

    s0: np.ndarray
    fit: PowerLawSpectrumFit | None = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), s0=self.s0)
        meta = {}
        if self.fit is not None:
            meta["fit"] = {
                "C": self.fit.C,
                "k0_sq": self.fit.k0_sq,
                "exponent": self.fit.exponent,
            }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> SkildSpectrum:
        path = Path(path)
        npz = path if path.suffix == ".npz" else path.with_suffix(".npz")
        s0 = np.load(npz)["s0"]
        meta_path = path.with_suffix(".json")
        fit = None
        if meta_path.is_file():
            raw = json.loads(meta_path.read_text())
            if "fit" in raw:
                f = raw["fit"]
                fit = PowerLawSpectrumFit(C=f["C"], k0_sq=f["k0_sq"], exponent=f["exponent"])
        return cls(s0=s0, fit=fit)


def save_spectrum(spec: SkildSpectrum, path: Path) -> None:
    spec.save(path)


def load_spectrum(path: Path) -> SkildSpectrum:
    return SkildSpectrum.load(path)


def estimate_spectrum_from_images(
    images: list[np.ndarray],
    *,
    channel_wise: bool = True,
) -> SkildSpectrum:
    if not images:
        raise ValueError("No images")
    h, w = images[0].shape[:2]
    if channel_wise and images[0].ndim == 3:
        channels = []
        fits = []
        for c in range(3):
            ch_imgs = [im[..., c] for im in images]
            centers, var = radial_variance_spectrum(ch_imgs)
            fit = fit_power_law_spectrum(centers, var)
            channels.append(spectrum_map_from_fit(h, w, fit))
            fits.append(fit)
        s0 = np.stack(channels, axis=-1)
        return SkildSpectrum(s0=s0.astype(np.float32), fit=fits[0])
    centers, var = radial_variance_spectrum(images)
    fit = fit_power_law_spectrum(centers, var)
    s0 = spectrum_map_from_fit(h, w, fit)
    return SkildSpectrum(s0=s0, fit=fit)


def estimate_spectrum_from_dir(data_dir: Path, *, limit: int = 256, channel_wise: bool = True) -> SkildSpectrum:
    return estimate_spectrum_from_images(load_images_from_dir(data_dir, limit=limit), channel_wise=channel_wise)


def _schedule_for_resolution(h: int, w: int, cfg: SkildConfig) -> SkildSchedule:
    k = frequency_radius_grid(h, w)
    return SkildSchedule(cfg.schedule, k_grid=k)


def start_timestep_for_scale(
    scale_factor: float,
    h: int,
    w: int,
    *,
    cfg: SkildConfig | None = None,
    spectrum: SkildSpectrum | None = None,
) -> int:
    """Pick reverse start step so effective resolution ≈ ``min(H,W)/scale_factor``."""
    cfg = cfg or SkildConfig()
    target_side = max(2.0, min(h, w) / max(scale_factor, 1.0))
    sched = _schedule_for_resolution(h, w, cfg)
    snr_thr = cfg.snr_threshold

    best = 0
    best_dist = float("inf")
    for step in range(cfg.schedule.num_steps):
        snr = sched.snr(step)
        k = sched.k_grid
        alive = snr >= snr_thr
        if not np.any(alive):
            eff_side = 1.0
        else:
            k_min = float(k[alive].min()) + 1e-6
            eff_side = np.pi / k_min
        dist = abs(eff_side - target_side)
        if dist < best_dist:
            best_dist = dist
            best = step
    return best


def _oracle_eps_predictor(xn: np.ndarray, x0_dct: np.ndarray, spectrum: np.ndarray, schedule: SkildSchedule, step: int):
    """Oracle ε for testing forward/reverse algebra (not for production)."""
    ab = schedule.alpha_bar(step)
    sqrt_one = np.sqrt(np.maximum(1.0 - ab, 0.0))
    s0_sqrt = np.sqrt(np.maximum(spectrum, 1e-12))
    return ((xn - np.sqrt(np.maximum(ab, 0.0)) * x0_dct) / (sqrt_one * s0_sqrt + 1e-12)).astype(np.float32)


def _run_reverse(
    x_start: np.ndarray,
    spectrum: np.ndarray,
    schedule: SkildSchedule,
    start_step: int,
    eps_fn: Callable[[np.ndarray, int], np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    x = x_start.astype(np.float32)
    for step in range(start_step, -1, -1):
        eps_pred = eps_fn(x, step)
        x = reverse_step_dct(x, eps_pred, spectrum, schedule, step, rng)
    return x


def continuous_super_resolve(
    image: np.ndarray,
    *,
    scale_factor: float = 4.0,
    spectrum: SkildSpectrum,
    cfg: SkildConfig | None = None,
    eps_predictor: Callable[[np.ndarray, int], np.ndarray] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Super-resolve ``image`` (H×W×3 in [0,1]) via partial SKILD reverse.

    Without a trained ``eps_predictor``, uses a **frequency soft-unmask** baseline:
    runs the exact forward marginal to ``start_step`` then applies oracle-free
    spectral denoising (high-frequency energy restoration from coarse DCT). Pass a
    checkpoint-backed predictor when available (see ``documents/SKILD.md``).
    """
    cfg = cfg or SkildConfig()
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    sched = _schedule_for_resolution(h, w, cfg)
    s0 = spectrum.s0
    if s0.shape[:2] != (h, w):
        raise ValueError(f"Spectrum shape {s0.shape[:2]} != image {(h, w)}")

    x0 = image_to_dct(image)
    n0 = start_timestep_for_scale(scale_factor, h, w, cfg=cfg, spectrum=spectrum)
    xn, _eps = forward_marginal_dct(x0, s0, sched, n0, rng)

    if eps_predictor is not None:
        x_dct = _run_reverse(xn, s0, sched, n0, eps_predictor, rng)
        out = dct_to_image(x_dct)
        return np.clip(out, 0.0, 1.0)

    # Baseline: Wiener-like gain on high-frequency modes using schedule SNR
    ab = sched.alpha_bar(n0)
    snr = sched.snr(n0)
    from ltx_trainer.skild.diffusion import _broadcast_schedule  # noqa: PLC0415

    snr_b = _broadcast_schedule(snr, xn)
    ab_b = _broadcast_schedule(ab, xn)
    gain = np.where(snr_b >= cfg.snr_threshold, 1.0, ab_b / np.maximum(1.0 - ab_b, 1e-6))
    x_restored = (xn * gain).astype(np.float32)
    out = dct_to_image(x_restored)
    return np.clip(out, 0.0, 1.0)


def generation_from_noise(
    shape: tuple[int, int, int],
    *,
    spectrum: SkildSpectrum,
    cfg: SkildConfig | None = None,
    eps_predictor: Callable[[np.ndarray, int], np.ndarray] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Unconditional sample from pure noise marginal at ``t = T-1`` (requires trained net)."""
    cfg = cfg or SkildConfig()
    h, w, c = shape
    rng = np.random.default_rng(seed)
    sched = _schedule_for_resolution(h, w, cfg)
    s0 = spectrum.s0
    n_last = cfg.schedule.num_steps - 1
    eps = rng.standard_normal((h, w) if c == 1 else (h, w, c)).astype(np.float32)
    noise_scale = np.sqrt(np.maximum(s0, 1e-12))
    xn = noise_scale * eps

    if eps_predictor is None:
        raise ValueError("generation_from_noise requires a trained ε-predictor checkpoint")

    x_dct = _run_reverse(xn, s0, sched, n_last, eps_predictor, rng)
    return np.clip(dct_to_image(x_dct), 0.0, 1.0)
