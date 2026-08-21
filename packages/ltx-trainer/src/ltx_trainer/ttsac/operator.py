r"""Generator–encoder composition and TT-SAC test-time refinement (Sec. III, Eq. 1–5).

Notation (paper): reference image I_r, audio A = {a_t}, identity encoder E, generator G.
Static conditioning f_r = E(I_r). Generator–encoder composition (Eq. 1):

    (E ∘ G)(f, A)_t := E(G(f, A)_t).

Monte Carlo estimate of the conditioning operator T(f) = E_t[(E ∘ G)(f, A)_t] (Eq. 2–3):

    \hat{T}(f_r) = (1/K) Σ_{t=1}^K (E ∘ G)(f_r, A)_t.

Single refinement step (Eq. 5): f_r ← \bar{f} with \bar{f} = \hat{T}(f_r), optionally blended
with the original embedding for stability (not in the paper; default 1.0 = full update).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

EncodeFn = Callable[[Tensor], Tensor]
GenerateFn = Callable[..., Tensor]


def generator_encoder_compose(
    f: Tensor,
    encode_fn: EncodeFn,
    generate_fn: GenerateFn,
    audio: Any,
    t: int,
) -> Tensor:
    """One step of (E ∘ G)(f, A)_t — encode the t-th frame produced by G under conditioning f."""
    frame = generate_fn(f, audio, t) if audio is not None else generate_fn(f, t)
    return encode_fn(frame)


def encode_generated_frames(
    f: Tensor,
    encode_fn: EncodeFn,
    generate_fn: GenerateFn,
    audio: Any = None,
    *,
    K: int,
) -> Tensor:
    """Stack K identity features (E ∘ G)(f, A)_t for t = 0 … K−1 (Eq. 3)."""
    feats: list[Tensor] = []
    for t in range(K):
        feats.append(generator_encoder_compose(f, encode_fn, generate_fn, audio, t))
    return torch.stack(feats, dim=0)


def monte_carlo_conditioning(
    f_ref: Tensor,
    encode_fn: EncodeFn,
    generate_fn: GenerateFn,
    audio: Any = None,
    *,
    K: int,
) -> Tensor:
    r"""Empirical conditioning operator ``\hat{T}(f_r)`` — row mean of stacked (E∘G) features (Eq. 3–4)."""
    stacked = encode_generated_frames(f_ref, encode_fn, generate_fn, audio, K=K)
    return stacked.mean(dim=0)


def refine_conditioning(
    f_ref: Tensor,
    encode_fn: EncodeFn,
    generate_fn: GenerateFn,
    audio: Any = None,
    *,
    K: int,
    mix: float = 1.0,
) -> Tensor:
    r"""Single fixed-point step. Paper: ``f_r ← \hat{T}(f_r)``. Optional convex blend with prior ``f_r``.

    ``mix=1`` reproduces Eq. (5). ``mix∈(0,1)`` interpolates ``(1-mix)·f_ref + mix·\bar{f}`` (heuristic stabilization).
    """
    f_bar = monte_carlo_conditioning(f_ref, encode_fn, generate_fn, audio, K=K)
    if mix >= 1.0:
        return f_bar
    if mix <= 0.0:
        return f_ref
    return (1.0 - mix) * f_ref + mix * f_bar


def apply_tt_sac(
    f_ref: Tensor,
    encode_fn: EncodeFn,
    generate_fn: GenerateFn,
    audio: Any = None,
    *,
    K: int = 3,
    f_motion: Tensor | None = None,
    mix: float = 1.0,
    motion_mix: float = 0.5,
) -> dict[str, Tensor]:
    """Two-pass TT-SAC: aggregate identity (and optionally motion) from early generated latents.

    ``audio`` is the paper's driving signal A (passed through to ``generate_fn`` when non-``None``).
    Motion pathway (paper ``+ TT-SAC (w/ motion)``): optional convex update of a separate motion vector.
    """
    f_identity = refine_conditioning(f_ref, encode_fn, generate_fn, audio, K=K, mix=mix)
    out: dict[str, Tensor] = {
        "f_initial": f_ref.detach().clone(),
        "f_refined": f_identity.detach().clone(),
    }
    if f_motion is not None:
        f_m_bar = monte_carlo_conditioning(f_motion, encode_fn, generate_fn, audio, K=K)
        out["f_motion_refined"] = ((1.0 - motion_mix) * f_motion + motion_mix * f_m_bar).detach().clone()
    return out
