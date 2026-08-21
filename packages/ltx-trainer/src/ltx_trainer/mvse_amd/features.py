"""Modality-detection feature vectors (arXiv:2606.05931)."""

from __future__ import annotations

import numpy as np


def modality_detection_features(
    ss: np.ndarray,
    sf: np.ndarray,
    cs_to_f: np.ndarray,
    cf_to_s: np.ndarray,
) -> np.ndarray:
    """Eq. (8): f = [ss; sf; cs→f; cf→s; μ; σ] ∈ R^{4n+8}."""
    mu = np.array([ss.mean(), sf.mean(), cs_to_f.mean(), cf_to_s.mean()])
    sigma = np.array([ss.std(), sf.std(), cs_to_f.std(), cf_to_s.std()])
    return np.concatenate([ss, sf, cs_to_f, cf_to_s, mu, sigma])
