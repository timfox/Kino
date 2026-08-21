"""ControlNet-style latent memory conditioning stub (Sec. 4.3, Appendix C)."""
from __future__ import annotations

import numpy as np

from ltx_trainer.mirage.rotary import FrameRole, apply_segment_rotary, frame_role_phases, segment_rotary_phases


def pack_control_input(
    z_hat: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    """Concatenate readout latent and visibility mask for side branch input."""
    z = np.asarray(z_hat, dtype=np.float64)
    m = np.asarray(visibility, dtype=np.float64)
    if m.ndim == 2:
        m = m[None, ...]
    if z.shape[1:] != m.shape[1:]:
        raise ValueError(f"shape mismatch z={z.shape} mask={m.shape}")
    return np.concatenate([z, m], axis=0)


def inject_side_branch(
    backbone_hidden: np.ndarray,
    control_hidden: np.ndarray,
    *,
    layer_idx: int,
    control_layers: tuple[int, ...],
    scale: float = 1.0,
) -> np.ndarray:
    """Add ControlNet-style residuals at selected transformer blocks."""
    out = np.asarray(backbone_hidden, dtype=np.float64)
    if layer_idx in control_layers:
        out = out + scale * np.asarray(control_hidden, dtype=np.float64)
    return out


def side_branch_forward(
    control_input: np.ndarray,
    *,
    hidden_dim: int = 3072,
    seed: int = 0,
) -> np.ndarray:
    """Toy side branch: pool readout and project to hidden dim (CPU stub)."""
    rng = np.random.default_rng(seed)
    z = np.asarray(control_input, dtype=np.float64)
    pooled = z.reshape(z.shape[0], -1).mean(axis=1)
    w = rng.standard_normal((hidden_dim, pooled.size)) * 0.01
    return w @ pooled


def segment_aware_attention_stub(
    q: np.ndarray,
    k: np.ndarray,
    visibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment-aware rotary on Q/K before cross-attention (Sec. 4.3)."""
    phases = segment_rotary_phases(visibility)
    return apply_segment_rotary(q, k, phases)


def frame_role_attention_stub(
    q: np.ndarray,
    k: np.ndarray,
    roles: list[FrameRole],
) -> tuple[np.ndarray, np.ndarray]:
    """Rotary offsets for noisy target / clean preceding / clean reference (Appendix C)."""
    phases = frame_role_phases(roles)
    while phases.ndim < q.ndim - 1:
        phases = phases[..., None]
    phases = np.broadcast_to(phases, q.shape[:-1])
    return apply_segment_rotary(q, k, phases)
