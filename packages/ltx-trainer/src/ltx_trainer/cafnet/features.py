"""MFCC, LFCC, and Chroma-STFT feature extraction proxies (Table 1)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cafnet.config import CafNetConfig


def _linear_filterbank(n_fft: int, n_filters: int, sr: int) -> np.ndarray:
    """Linear-spaced filterbank for LFCC (not Mel-warped)."""
    freqs = np.linspace(0, sr / 2, n_filters + 2)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_filters):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left or right == center:
            continue
        for k in range(left, center):
            fb[i, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[i, k] = (right - k) / max(right - center, 1)
    return fb


def _mel_filterbank(n_fft: int, n_filters: int, sr: int) -> np.ndarray:
    """Triangular Mel filterbank for MFCC."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_filters + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_filters):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left or right == center:
            continue
        for k in range(left, center):
            fb[i, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[i, k] = (right - k) / max(right - center, 1)
    return fb


def _pad_frames(mat: np.ndarray, target: int) -> np.ndarray:
    c, t = mat.shape
    if t >= target:
        return mat[:, :target]
    return np.pad(mat, ((0, 0), (0, target - t)), mode="edge")


def stft_power(wave: np.ndarray, *, n_fft: int = 512, hop: int = 256) -> np.ndarray:
    """Short-time power spectrum (frames × bins)."""
    x = np.asarray(wave, dtype=np.float64)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    frames = 1 + max(0, (x.size - n_fft) // hop)
    out = np.zeros((frames, n_fft // 2 + 1), dtype=np.float64)
    window = np.hanning(n_fft)
    for i in range(frames):
        start = i * hop
        chunk = x[start : start + n_fft]
        if chunk.size < n_fft:
            chunk = np.pad(chunk, (0, n_fft - chunk.size))
        spec = np.fft.rfft(chunk * window)
        out[i] = np.abs(spec) ** 2
    return out


def extract_mfcc(wave: np.ndarray, cfg: CafNetConfig | None = None) -> np.ndarray:
    """MFCC matrix (coeffs × frames)."""
    cfg = cfg or CafNetConfig()
    power = stft_power(wave).T  # bins × frames
    fb = _mel_filterbank(512, cfg.mfcc_coeffs, cfg.sample_rate_hz)
    mel = fb @ power
    log_mel = np.log(np.maximum(mel, 1e-10))
    # DCT-like cepstral projection (orthogonal basis proxy)
    t = np.arange(cfg.mfcc_coeffs)[:, None]
    k = np.arange(cfg.mfcc_coeffs)[None, :]
    dct = np.cos(np.pi * (2 * k + 1) * t / (2 * cfg.mfcc_coeffs))
    return _pad_frames(dct @ log_mel, cfg.feature_frames)


def extract_lfcc(wave: np.ndarray, cfg: CafNetConfig | None = None) -> np.ndarray:
    """LFCC with explicit linear filterbank (coeffs × frames)."""
    cfg = cfg or CafNetConfig()
    power = stft_power(wave).T
    fb = _linear_filterbank(512, cfg.lfcc_coeffs, cfg.sample_rate_hz)
    lin = fb @ power
    log_lin = np.log(np.maximum(lin, 1e-10))
    t = np.arange(cfg.lfcc_coeffs)[:, None]
    k = np.arange(cfg.lfcc_coeffs)[None, :]
    dct = np.cos(np.pi * (2 * k + 1) * t / (2 * cfg.lfcc_coeffs))
    return _pad_frames(dct @ log_lin, cfg.feature_frames)


def extract_chroma(wave: np.ndarray, cfg: CafNetConfig | None = None) -> np.ndarray:
    """Chroma-STFT (12 × frames) from power spectrum folding."""
    cfg = cfg or CafNetConfig()
    power = stft_power(wave)  # frames × bins
    n_bins = power.shape[1]
    chroma = np.zeros((cfg.chroma_bins, power.shape[0]), dtype=np.float64)
    for b in range(n_bins):
        pitch_class = b % cfg.chroma_bins
        chroma[pitch_class] += power[:, b]
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)
    return _pad_frames(chroma, cfg.feature_frames)


def extract_feature_triplet(wave: np.ndarray, cfg: CafNetConfig | None = None) -> dict[str, np.ndarray]:
    """Parallel MFCC, LFCC, Chroma per Table 1."""
    cfg = cfg or CafNetConfig()
    return {
        "mfcc": extract_mfcc(wave, cfg),
        "lfcc": extract_lfcc(wave, cfg),
        "chroma": extract_chroma(wave, cfg),
    }


def feature_shapes(cfg: CafNetConfig | None = None) -> dict[str, tuple[int, int]]:
    cfg = cfg or CafNetConfig()
    return {
        "mfcc": (cfg.mfcc_coeffs, cfg.feature_frames),
        "lfcc": (cfg.lfcc_coeffs, cfg.feature_frames),
        "chroma": (cfg.chroma_bins, cfg.feature_frames),
    }
