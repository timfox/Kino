"""Oklch+ L/C transforms and color-difference models (Eqs. 3–6)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ltx_trainer.oklch_plus.config import OklchPlusParams, PowerLCParams
from ltx_trainer.oklch_plus.oklab import oklab_to_oklch, oklch_to_oklab


def power_lightness(l: np.ndarray, *, alpha: float) -> np.ndarray:
    l = np.maximum(np.asarray(l, dtype=np.float64), 0.0)
    return l**alpha


def naka_rushton_chroma(c: np.ndarray, *, n: float, sigma: float) -> np.ndarray:
    c = np.maximum(np.asarray(c, dtype=np.float64), 0.0)
    cn = c**n
    sn = sigma**n
    return cn / (cn + sn)


def power_chroma(c: np.ndarray, *, gamma: float) -> np.ndarray:
    c = np.maximum(np.asarray(c, dtype=np.float64), 0.0)
    return c**gamma


def log_chroma(c: np.ndarray, *, beta: float) -> np.ndarray:
    c = np.maximum(np.asarray(c, dtype=np.float64), 0.0)
    return np.log1p(beta * c)


def sigmoid_chroma(c: np.ndarray, *, a: float, cth: float) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-a * (c - cth)))


def transform_lab(
    lab: np.ndarray,
    *,
    l_fn: Callable[[np.ndarray], np.ndarray],
    c_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    l_, c, h = oklab_to_oklch(lab)
    lp = l_fn(l_)
    cp = c_fn(c)
    return oklch_to_oklab(lp, cp, h)


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    d = np.asarray(lab1, dtype=np.float64) - np.asarray(lab2, dtype=np.float64)
    return float(np.sqrt(np.sum(d**2)))


def delta_e_oklab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return delta_e(lab1, lab2)


def delta_e_oklch_plus(
    lab1: np.ndarray,
    lab2: np.ndarray,
    *,
    params: OklchPlusParams | None = None,
) -> float:
    p = params or OklchPlusParams()
    t1 = transform_lab(
        lab1,
        l_fn=lambda x: power_lightness(x, alpha=p.alpha),
        c_fn=lambda x: naka_rushton_chroma(x, n=p.n, sigma=p.sigma),
    )
    t2 = transform_lab(
        lab2,
        l_fn=lambda x: power_lightness(x, alpha=p.alpha),
        c_fn=lambda x: naka_rushton_chroma(x, n=p.n, sigma=p.sigma),
    )
    return delta_e(t1, t2)


def delta_e_power_lc(
    lab1: np.ndarray,
    lab2: np.ndarray,
    *,
    params: PowerLCParams | None = None,
) -> float:
    p = params or PowerLCParams()
    t1 = transform_lab(
        lab1,
        l_fn=lambda x: power_lightness(x, alpha=p.alpha),
        c_fn=lambda x: power_chroma(x, gamma=p.gamma),
    )
    t2 = transform_lab(
        lab2,
        l_fn=lambda x: power_lightness(x, alpha=p.alpha),
        c_fn=lambda x: power_chroma(x, gamma=p.gamma),
    )
    return delta_e(t1, t2)


def delta_e_with_chroma_fn(
    lab1: np.ndarray,
    lab2: np.ndarray,
    *,
    alpha: float,
    c_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    t1 = transform_lab(lab1, l_fn=lambda x: power_lightness(x, alpha=alpha), c_fn=c_fn)
    t2 = transform_lab(lab2, l_fn=lambda x: power_lightness(x, alpha=alpha), c_fn=c_fn)
    return delta_e(t1, t2)


def interpolate_oklch_plus(lab_a: np.ndarray, lab_b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation in transformed (L′, a′, b′) space (§6)."""
    p = OklchPlusParams()
    ta = transform_lab(
        lab_a,
        l_fn=lambda x: power_lightness(x, alpha=p.alpha),
        c_fn=lambda x: naka_rushton_chroma(x, n=p.n, sigma=p.sigma),
    )
    tb = transform_lab(
        lab_b,
        l_fn=lambda x: power_lightness(x, alpha=p.alpha),
        c_fn=lambda x: naka_rushton_chroma(x, n=p.n, sigma=p.sigma),
    )
    return ta + float(t) * (tb - ta)
