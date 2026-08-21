"""Synthetic security-critical HSI and AHAD evaluation demos."""

from __future__ import annotations

import numpy as np

from ltx_trainer.ahad.config import AHADConfig, AHADParams
from ltx_trainer.ahad.criterion import estimate_robust_perturbation
from ltx_trainer.ahad.hsi_ops import shift_set, shift_tensor
from ltx_trainer.ahad.metrics import armcba, auc_pd_pf, glf_denoise_proxy, rx_anomaly_scores


def synthetic_hsi(
    size: int = 48,
    bands: int = 16,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth background + sparse planted anomalies."""
    rng = np.random.default_rng(seed)
    h = w = size
    x = np.linspace(0, 1, bands)
    base = np.stack([np.sin(2 * np.pi * x + k) for k in np.linspace(0, 1.5, h * w)], axis=0)
    y = base.reshape(h, w, bands)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    mask = np.zeros((h, w), dtype=bool)
    cx, cy = h // 3, w // 3
    mask[cx - 2 : cx + 3, cy - 2 : cy + 3] = True
    mask[h // 2 : h // 2 + 2, w // 2 : w // 2 + 2] = True
    anomaly_sig = rng.normal(0.0, 0.15, size=(bands,))
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                y[i, j, :] = np.clip(y[i, j, :] + anomaly_sig, 0.0, 1.0)
    return y.astype(np.float64), mask


def crop_surrogate(y: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return y.copy()
    if y.ndim == 2:
        return y[radius:-radius, radius:-radius].copy()
    return y[radius:-radius, radius:-radius, :].copy()


def evaluate_ahad_on_hsi(
    y_surrogate: np.ndarray,
    mask: np.ndarray,
    params: AHADParams,
    *,
    use_sstv: bool = True,
    use_lipschitz: bool = True,
    use_pag: bool = True,
    denoise: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    p = estimate_robust_perturbation(
        y_surrogate,
        params,
        use_sstv=use_sstv,
        use_lipschitz=use_lipschitz,
        use_pag=use_pag,
        rng=rng,
    )
    y_a = np.clip(y_surrogate + p, 0.0, 1.0)
    if denoise:
        y_a = glf_denoise_proxy(y_a)
    scores_up = rx_anomaly_scores(y_surrogate, subspace_dim=params.subspace_dim)
    scores_ap = rx_anomaly_scores(y_a, subspace_dim=params.subspace_dim)
    auc_up = auc_pd_pf(scores_up, mask)
    auc_ap = auc_pd_pf(scores_ap, mask)
    return {
        "auc_up": auc_up,
        "auc_ap": auc_ap,
        "armcba": armcba(auc_up, auc_ap),
        "perturbation_energy": float(np.sqrt(np.sum(p**2))),
    }


def shift_robustness_demo(
    y_surrogate: np.ndarray,
    mask: np.ndarray,
    p_star: np.ndarray,
    params: AHADParams,
) -> dict[str, float]:
    armcbas: list[float] = []
    for delta in shift_set(params.shift_radius):
        y_shift = shift_tensor(y_surrogate, delta)
        m_shift = shift_tensor(mask.astype(np.float64), delta) > 0.5
        y_a = np.clip(y_shift + p_star, 0.0, 1.0)
        auc_up = auc_pd_pf(rx_anomaly_scores(y_shift, subspace_dim=params.subspace_dim), m_shift)
        auc_ap = auc_pd_pf(rx_anomaly_scores(y_a, subspace_dim=params.subspace_dim), m_shift)
        armcbas.append(armcba(auc_up, auc_ap))
    return {"mean_armcba": float(np.mean(armcbas)), "min_armcba": float(np.min(armcbas))}


def ablation_table(config: AHADConfig | None = None, *, seed: int = 0) -> list[dict[str, object]]:
    cfg = config or AHADConfig()
    params = AHADParams(iterations=20)
    y_full, mask_full = synthetic_hsi(seed=seed)
    r = params.shift_radius
    y = crop_surrogate(y_full, r)
    mask = crop_surrogate(mask_full.astype(np.float64), r) > 0.5
    cases = [
        ("Lipschitz-PAG", False, True, True, False),
        ("SSTV-Lipschitz", True, True, False, False),
        ("SSTV-PAG", True, False, True, False),
        ("SSTV-Lipschitz-PAG", True, True, True, False),
        ("SSTV-Lipschitz-PAG+GLF", True, True, True, True),
    ]
    rows: list[dict[str, object]] = []
    for name, sstv, lip, pag, den in cases:
        row = evaluate_ahad_on_hsi(
            y,
            mask,
            params,
            use_sstv=sstv,
            use_lipschitz=lip,
            use_pag=pag,
            denoise=den,
            rng=np.random.default_rng(seed),
        )
        rows.append({"case": name, **row})
    return rows


