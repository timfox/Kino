"""STFT/iSTFT scriptability + inverse guard (Sec. 6.1–6.2)."""

from __future__ import annotations

SUPPORTED_ISTFT_FREQ_SCALES = frozenset({"no"})
UNSUPPORTED_ISTFT_FREQ_SCALES = frozenset({"linear", "log", "log2"})


class IstftFreqScaleError(RuntimeError):
    """Raised when inverse STFT is requested on a non-uniform frequency grid."""


def validate_istft_freq_scale(freq_scale: str) -> None:
    if freq_scale not in SUPPORTED_ISTFT_FREQ_SCALES:
        raise IstftFreqScaleError(
            f"Reliable nnAudio iSTFT inversion requires freq_scale='no'; got {freq_scale!r}. "
            "Use Griffin–Lim on a linear-frequency spectrogram or a log-frequency frame method."
        )


def overlap_add_length(length: int | None, tensor_len: int) -> int:
    """Appendix-style Optional[int] narrowing for scripted helpers."""
    if length is None:
        return tensor_len
    return int(length)


def torchscript_fixes() -> list[dict[str, str]]:
    return [
        {
            "issue": "dynamic state mutation in forward",
            "fix": "local num_samples + local w_sum; eager cache gated on torch.jit.is_scripting()",
        },
        {
            "issue": "dynamic padding submodules in forward",
            "fix": "torch.nn.functional.pad with mode string",
        },
        {
            "issue": "unnarrowed Optional[int] / x == None",
            "fix": "explicit None branch then int narrowing; use `is None`",
        },
    ]


def round_trip_error_uniform(x: list[float], y: list[float]) -> float:
    """Mean absolute error for supported freq_scale='no' round-trip stub."""
    if not x:
        return 0.0
    return sum(abs(a - b) for a, b in zip(x, y, strict=True)) / len(x)
