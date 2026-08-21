"""Synthetic COMBVD-like pairs and model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ltx_trainer.oklch_plus.config import OklchPlusParams, PowerLCParams
from ltx_trainer.oklch_plus.metrics import (
    table1_chroma_functions,
    table2_overall_stress,
    table3_subdataset_stress,
    table5_cross_validation,
)
from ltx_trainer.oklch_plus.oklab import oklab_to_oklch, srgb_to_oklab
from ltx_trainer.oklch_plus.stress import stress
from ltx_trainer.oklch_plus.transforms import (
    delta_e,
    delta_e_oklab,
    delta_e_oklch_plus,
    delta_e_power_lc,
    delta_e_with_chroma_fn,
    log_chroma,
    naka_rushton_chroma,
    power_chroma,
    sigmoid_chroma,
)


@dataclass(frozen=True)
class ColorPair:
    lab1: np.ndarray
    lab2: np.ndarray
    delta_v: float


def _random_oklab_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Low-to-mid chroma samples (COMBVD-like: most C < 0.20)."""
    l1 = rng.uniform(0.15, 0.92)
    c1 = abs(rng.normal(0.0, 0.08))
    h1 = rng.uniform(-np.pi, np.pi)
    l2 = np.clip(l1 + rng.normal(0.0, 0.12), 0.05, 0.98)
    c2 = np.clip(c1 + abs(rng.normal(0.0, 0.06)), 0.0, 0.35)
    h2 = h1 + rng.normal(0.0, 0.35)
    a1, b1 = c1 * np.cos(h1), c1 * np.sin(h1)
    a2, b2 = c2 * np.cos(h2), c2 * np.sin(h2)
    return np.array([l1, a1, b1]), np.array([l2, a2, b2])


def synthetic_combvd_pairs(*, n: int = 256, seed: int = 42) -> list[ColorPair]:
    """Pairs with ΔV aligned to Oklch+ distance (validates STRESS pipeline)."""
    rng = np.random.default_rng(seed)
    pairs: list[ColorPair] = []
    params = OklchPlusParams()
    for _ in range(n):
        lab1, lab2 = _random_oklab_pair(rng)
        de = delta_e_oklch_plus(lab1, lab2, params=params)
        noise = 1.0 + 0.08 * rng.normal()
        pairs.append(ColorPair(lab1=lab1, lab2=lab2, delta_v=max(de * noise, 1e-6)))
    return pairs


def evaluate_delta_fn(
    pairs: list[ColorPair],
    delta_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    de = np.array([delta_fn(p.lab1, p.lab2) for p in pairs])
    dv = np.array([p.delta_v for p in pairs])
    return stress(de, dv)


def synthetic_stress_comparison(*, n: int = 256, seed: int = 42) -> dict[str, float]:
    pairs = synthetic_combvd_pairs(n=n, seed=seed)
    return {
        "oklab": evaluate_delta_fn(pairs, delta_e_oklab),
        "oklch_plus": evaluate_delta_fn(pairs, delta_e_oklch_plus),
        "power_lc": evaluate_delta_fn(pairs, delta_e_power_lc),
    }


def chroma_function_stress_on_synthetic(*, n: int = 256, seed: int = 7) -> dict[str, float]:
    """Rank chroma transforms on synthetic pairs (directional vs Table 1)."""
    rng = np.random.default_rng(seed)
    pairs: list[ColorPair] = []
    alpha = 0.73
    for _ in range(n):
        lab1, lab2 = _random_oklab_pair(rng)
        pairs.append(ColorPair(lab1=lab1, lab2=lab2, delta_v=1.0))  # placeholder

    def run(c_fn: Callable[[np.ndarray], np.ndarray]) -> float:
        de = np.array([delta_e_with_chroma_fn(p.lab1, p.lab2, alpha=alpha, c_fn=c_fn) for p in pairs])
        dv = np.array(
            [
                delta_e_with_chroma_fn(
                    p.lab1,
                    p.lab2,
                    alpha=alpha,
                    c_fn=lambda x: naka_rushton_chroma(x, n=0.87, sigma=0.34),
                )
                for p in pairs
            ]
        )
        return stress(de, dv)

    p = OklchPlusParams()
    pl = PowerLCParams()
    return {
        "power": run(lambda x: power_chroma(x, gamma=pl.gamma)),
        "log": run(lambda x: log_chroma(x, beta=4.92)),
        "sigmoid": run(lambda x: sigmoid_chroma(x, a=42.0, cth=0.068)),
        "naka_rushton": run(lambda x: naka_rushton_chroma(x, n=p.n, sigma=p.sigma)),
    }


def combvd_low_chroma_fraction(pairs: list[ColorPair]) -> float:
    cs = []
    for p in pairs:
        _, c1, _ = oklab_to_oklch(p.lab1)
        _, c2, _ = oklab_to_oklch(p.lab2)
        cs.extend([float(c1), float(c2)])
    return float(np.mean(np.array(cs) < 0.20))


def model_comparison() -> dict[str, float]:
    """Paper anchor STRESS (Table 2)."""
    t2 = table2_overall_stress()
    return {k: float(v["stress"]) for k, v in t2.items() if k != "helmlab_ref"}


def subdataset_improvement() -> dict[str, float]:
    """Oklab − Oklch+ STRESS per sub-dataset (Table 3)."""
    t3 = table3_subdataset_stress()
    return {k: v["oklab"] - v["oklch_plus"] for k, v in t3.items()}


def interpolation_uniformity_demo(*, steps: int = 9) -> dict[str, float]:
    """Step-size coefficient of variation along interpolation path (lower ⇒ more uniform)."""
    lab_a = np.array([0.62, 0.08, 0.04])
    lab_b = np.array([0.38, -0.06, 0.10])
    ts = np.linspace(0.0, 1.0, steps)

    oklab_pts = [lab_a + t * (lab_b - lab_a) for t in ts]
    oklab_steps = [
        delta_e_oklab(oklab_pts[i - 1], oklab_pts[i]) for i in range(1, len(oklab_pts))
    ]

    from ltx_trainer.oklch_plus.transforms import interpolate_oklch_plus

    plus_pts = [interpolate_oklch_plus(lab_a, lab_b, t) for t in ts]
    plus_steps = [delta_e(plus_pts[i - 1], plus_pts[i]) for i in range(1, len(plus_pts))]

    def coeff_var(steps: list[float]) -> float:
        arr = np.asarray(steps, dtype=np.float64)
        return float(np.std(arr) / (np.mean(arr) + 1e-9))

    return {
        "oklab_step_cv": coeff_var(oklab_steps),
        "oklch_plus_step_cv": coeff_var(plus_steps),
        "oklab_midpoint_ratio": coeff_var(oklab_steps),
        "oklch_plus_midpoint_ratio": coeff_var(plus_steps),
    }


def cross_validation_summary() -> dict[str, float]:
    t5 = table5_cross_validation()
    test = t5["test_bfd_p_d65"]
    return {
        "test_oklab": test["oklab"],
        "test_oklch_plus": test["oklch_plus_cv"],
        "test_ciede2000": test["ciede2000_ref"],
        "train_oklch_plus": t5["train_oklch_plus_cv"],
    }


def table1_vs_nr_advantage() -> float:
    t1 = table1_chroma_functions()
    return float(t1["power"]["stress"] - t1["naka_rushton"]["stress"])
