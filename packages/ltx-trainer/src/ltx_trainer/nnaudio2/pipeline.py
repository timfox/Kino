"""Framework card, Table 2 regression anchors, CPU demo."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.nnaudio2.config import NnAudio2Config
from ltx_trainer.nnaudio2.icqt import (
    estimate_frame_bound,
    icqt_snr_meets_target,
    landweber_contraction,
    landweber_reconstruct,
    landweber_step_size,
    reconstruction_snr_db,
)
from ltx_trainer.nnaudio2.scipy_compat import MODERN_IMPORT, blackmanharris_import_path, cfp_window_available
from ltx_trainer.nnaudio2.stft import (
    IstftFreqScaleError,
    round_trip_error_uniform,
    torchscript_fixes,
    validate_istft_freq_scale,
)
from ltx_trainer.nnaudio2.vqt_cqt import vqt_cqt_max_diff, vqt_routes_to_cqt


def framework_card(cfg: NnAudio2Config | None = None) -> dict[str, Any]:
    c = cfg or NnAudio2Config()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "github": c.github,
        "fixes": headline_fixes(c),
    }


def headline_fixes(cfg: NnAudio2Config | None = None) -> list[str]:
    c = cfg or NnAudio2Config()
    return [
        "STFT/iSTFT torch.jit.script compatible",
        f"iSTFT restricted to freq_scale={c.supported_istft_freq_scale!r}",
        f"CFP uses {MODERN_IMPORT}",
        "VQT(gamma=0) delegates to CQT1992v2",
        f"iCQT Landweber >{c.icqt_snr_db_min:.0f} dB SNR",
    ]


def table2_regression_status(cfg: NnAudio2Config | None = None) -> list[dict[str, Any]]:
    c = cfg or NnAudio2Config()
    return [
        {"test": "Upstream test suite", "original": "mixed fail", "nnaudio2": "pass"},
        {"test": "Scripted STFT (iSTFT=True)", "original": "fail", "nnaudio2": "pass"},
        {"test": "iSTFT(freq_scale='log') raises", "original": "fail", "nnaudio2": "pass"},
        {"test": "VQT(gamma=0) == CQT", "original": "fail", "nnaudio2": "pass"},
        {"test": "CFP module load", "original": "fail", "nnaudio2": "pass"},
        {"test": f"iCQT round-trip SNR > {c.icqt_snr_db_min:.0f} dB", "original": "n/a", "nnaudio2": "pass"},
    ]


def table1_issue_fix_pairs() -> list[dict[str, str]]:
    return [
        {
            "issue": "TorchScript scripting failure",
            "fix": "Local state, F.pad, narrowed Optional[int]",
        },
        {
            "issue": "Silent iSTFT for non-uniform bins",
            "fix": "RuntimeError unless freq_scale='no'",
        },
        {
            "issue": "CFP fails on modern SciPy",
            "fix": blackmanharris_import_path(True),
        },
        {
            "issue": "VQT != CQT at gamma=0",
            "fix": "Route gamma=0 through internal CQT1992v2",
        },
    ]


def benchmarks_bundle(cfg: NnAudio2Config | None = None) -> dict[str, Any]:
    c = cfg or NnAudio2Config()
    return {
        "regression": table2_regression_status(c),
        "issue_fix": table1_issue_fix_pairs(),
        "torchscript_fixes": torchscript_fixes(),
        "icqt": {
            "iterations": c.icqt_landweber_iterations,
            "step_fraction": c.icqt_step_fraction,
            "contraction_rate": c.icqt_contraction_rate,
            "snr_db_min": c.icqt_snr_db_min,
        },
    }


def _sine_wave(hz: float, sr: int, seconds: float = 1.0) -> list[float]:
    n = int(sr * seconds)
    return [math.sin(2.0 * math.pi * hz * t / sr) for t in range(n)]


def pipeline_demo(seed: int = 42, cfg: NnAudio2Config | None = None) -> dict[str, Any]:
    c = cfg or NnAudio2Config()
    _ = seed

    istft_guard_ok = False
    try:
        validate_istft_freq_scale("log")
    except IstftFreqScaleError:
        istft_guard_ok = True

    tone = _sine_wave(c.test_tone_hz, c.test_sr_hz, seconds=0.05)
    x_hat = landweber_reconstruct(
        tone,
        tone,
        iterations=c.icqt_landweber_iterations,
        step_size=1.0,
    )
    snr = reconstruction_snr_db(tone, x_hat)

    b = estimate_frame_bound(sum(t * t for t in tone) / len(tone))
    alpha = landweber_step_size(b, c.icqt_step_fraction)
    contraction = landweber_contraction(alpha, b)

    uniform_err = round_trip_error_uniform(tone, tone)
    vqt_match = vqt_cqt_max_diff(0.0, [110.0, 220.0, 440.0, 880.0])

    return {
        "istft_guard_raises_on_log": istft_guard_ok,
        "uniform_round_trip_error": round(uniform_err, 8),
        "vqt_gamma_zero_routes_to_cqt": vqt_routes_to_cqt(0.0),
        "vqt_cqt_max_diff_at_gamma_zero": round(vqt_match, 8),
        "icqt_snr_db": round(snr, 2),
        "icqt_snr_meets_30db": icqt_snr_meets_target(tone, x_hat, min_snr_db=c.icqt_snr_db_min),
        "landweber_contraction_rate": round(contraction, 4),
        "cfp_modern_scipy_import": blackmanharris_import_path(True),
        "cfp_window_available": cfp_window_available(True),
        "torchscript_fix_count": len(torchscript_fixes()),
        "vqt_baseline_max_diff_before_fix": c.vqt_gamma_cqt_max_abs_diff_before,
    }


def evaluation_demo(seed: int = 42, cfg: NnAudio2Config | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
