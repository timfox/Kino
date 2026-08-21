"""Computed evaluation metrics (FAD/WER proxies, Dasheng AudioGen §4)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ltx_trainer.dasheng_audiogen.tokenizer import encode_waveform


def log_mel_proxy(wave: np.ndarray, sample_rate: float, *, n_mels: int = 64) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float64).ravel()
    hop = max(1, int(sample_rate / 100))
    n_frames = max(1, wave.size // hop)
    mel = np.zeros((n_frames, n_mels), dtype=np.float64)
    for i in range(n_frames):
        chunk = wave[i * hop : (i + 1) * hop]
        if chunk.size == 0:
            continue
        spec = np.abs(np.fft.rfft(chunk))
        bins = np.linspace(0, spec.size, n_mels + 1, dtype=int)
        for m in range(n_mels):
            seg = spec[bins[m] : bins[m + 1]]
            mel[i, m] = np.log1p(np.mean(seg) if seg.size else 0.0)
    return mel


def fad_proxy(ref_waves: Sequence[np.ndarray], gen_waves: Sequence[np.ndarray], *, sr: float = 48000.0) -> float:
    """Fréchet audio distance proxy: mean feature L2 between pooled embeddings."""
    def pool(waves: Sequence[np.ndarray]) -> np.ndarray:
        feats = [encode_waveform(w, sr).mean(axis=0) for w in waves]
        return np.stack(feats, axis=0)

    mu_ref = pool(ref_waves).mean(axis=0)
    mu_gen = pool(gen_waves).mean(axis=0)
    return float(np.linalg.norm(mu_ref - mu_gen))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein WER on whitespace tokens."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    for i in range(len(ref) + 1):
        dp[i, 0] = i
    for j in range(len(hyp) + 1):
        dp[0, j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[len(ref), len(hyp)] / len(ref))


def metrics_from_arrays(
    ref_waves: Sequence[np.ndarray],
    gen_waves: Sequence[np.ndarray],
    *,
    transcripts: Sequence[str] | None = None,
    asr_hyps: Sequence[str] | None = None,
    sr: float = 48000.0,
) -> dict[str, float]:
    fad = fad_proxy(ref_waves, gen_waves, sr=sr)
    out: dict[str, float] = {"fad_proxy": fad}
    if transcripts and asr_hyps and len(transcripts) == len(asr_hyps):
        wers = [word_error_rate(r, h) for r, h in zip(transcripts, asr_hyps)]
        out["wer_mean"] = float(np.mean(wers))
    return out


def metrics_compute_smoke() -> dict[str, Any]:
    sr = 48000.0
    t = np.arange(int(sr)) / sr
    ref = [np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)]
    gen = [np.sin(2 * np.pi * 445 * t), np.sin(2 * np.pi * 885 * t)]
    m = metrics_from_arrays(ref, gen, transcripts=["hello world"], asr_hyps=["hello word"])
    return {"fad_finite": np.isfinite(m["fad_proxy"]), "wer": m.get("wer_mean", 0.0)}
