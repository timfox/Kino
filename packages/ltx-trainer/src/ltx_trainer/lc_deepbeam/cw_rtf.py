"""Covariance-whitening (CW) RTF + interference subspace stubs (Eqs. 6–11)."""

from __future__ import annotations

import numpy as np


def noise_covariance(y_vk: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Eq. (6): Φ_nn(k) from noise-only frames.

    Args:
        y_vk: complex STFT snapshots shaped (V, M) for one frequency bin.
    """
    y = np.asarray(y_vk)
    if y.ndim != 2:
        raise ValueError("y_vk must be (V, M)")
    v = y.shape[0]
    if v == 0:
        return delta * np.eye(y.shape[1], dtype=np.complex128)
    phi = (y.conj().T @ y) / float(v)
    return phi + delta * np.eye(phi.shape[0], dtype=phi.dtype)


def inv_sqrtm_hermitian(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute A^{-1/2} for Hermitian PSD via EVD."""
    h = np.asarray(a)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("a must be square")
    # Force Hermitian
    h = 0.5 * (h + h.conj().T)
    w, v = np.linalg.eigh(h)
    w = np.maximum(w.real, 0.0)
    inv_sqrt = 1.0 / np.sqrt(w + eps)
    return (v * inv_sqrt) @ v.conj().T


def whiten(y_vk: np.ndarray, phi_nn: np.ndarray) -> np.ndarray:
    """Eq. (7): y_w = Φ_nn^{-1/2} y."""
    y = np.asarray(y_vk)
    w = inv_sqrtm_hermitian(phi_nn)
    return (w @ y.T).T


def whitened_covariance(y_vk: np.ndarray, phi_nn: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Eq. (9): Φ_ywyw = Φ_nn^{-1/2} Φ_yy (Φ_nn^{-1/2})^H."""
    y = np.asarray(y_vk)
    if y.ndim != 2:
        raise ValueError("y_vk must be (V, M)")
    v = y.shape[0]
    if v == 0:
        m = y.shape[1]
        return delta * np.eye(m, dtype=np.complex128)
    phi_yy = (y.conj().T @ y) / float(v) + delta * np.eye(y.shape[1], dtype=np.complex128)
    w = inv_sqrtm_hermitian(phi_nn)
    return w @ phi_yy @ w.conj().T


def rtf_from_dominant_eig(phi_ywyw: np.ndarray, phi_nn: np.ndarray, ref_mic: int = 0) -> np.ndarray:
    """Eq. (10): a(k) = Φ_nn^{H/2} ψ / (e_ref^T Φ_nn^{H/2} ψ).

    We obtain ψ as the dominant eigenvector of the whitened covariance.
    """
    w, v = np.linalg.eigh(0.5 * (phi_ywyw + phi_ywyw.conj().T))
    psi = v[:, int(np.argmax(w.real))]
    # Φ_nn^{H/2} = (Φ_nn^{1/2})^H; for Hermitian, equals Φ_nn^{1/2}
    phi_sqrt = np.linalg.cholesky(0.5 * (phi_nn + phi_nn.conj().T))
    a = phi_sqrt.conj().T @ psi
    denom = a[ref_mic]
    if abs(denom) < 1e-12:
        return a
    return a / denom


def interference_subspace_from_eigs(
    phi_ywyw: np.ndarray,
    phi_nn: np.ndarray,
    *,
    j_minus_one: int,
    ref_mic: int = 0,
) -> np.ndarray:
    """Eq. (11): basis vectors from the dominant (J−1) eigenspace."""
    if j_minus_one <= 0:
        raise ValueError("j_minus_one must be positive")
    w, v = np.linalg.eigh(0.5 * (phi_ywyw + phi_ywyw.conj().T))
    idx = np.argsort(w.real)[::-1][:j_minus_one]
    phi_sqrt = np.linalg.cholesky(0.5 * (phi_nn + phi_nn.conj().T))
    u_list: list[np.ndarray] = []
    for j in idx:
        uj = phi_sqrt.conj().T @ v[:, int(j)]
        denom = uj[ref_mic]
        if abs(denom) >= 1e-12:
            uj = uj / denom
        u_list.append(uj)
    return np.stack(u_list, axis=1)  # (M, J-1)

