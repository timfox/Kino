"""Whole shifted inverse — Algorithm 1 (Watt [38], Marchioro revisions)."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.midint_division.digits import precision_digits, shinv_direct, shift_n


def pow_diff(v: int, w: int, h: int, ell: int, base: int) -> tuple[bool, int]:
    """PowDiff(v, w, h, ℓ, B) → (sign positive iff B^h > v·w, magnitude)."""
    l_prec = precision_digits(v, base) + precision_digits(w, base) - ell + 1
    bh = pow(base, h)
    prod = v * w
    if v == 0 or w == 0 or l_prec >= h:
        if bh > prod:
            return True, bh - prod
        return False, prod - bh
    mod = pow(base, l_prec)
    p = prod % mod
    if p == 0:
        return True, 0
    if (p // pow(base, l_prec - 1)) == 0:
        return False, p
    return True, mod - p


def step(v: int, w: int, h: int, m: int, ell: int, g: int, base: int) -> int:
    """Single Newton Step with unsigned borrow guard."""
    sign, x = pow_diff(v, w, h - m, ell - g, base)
    if sign:
        return shift_n(w, m, base) + shift_n(w * x, 2 * m - h, base)
    tmp = w * x
    res = shift_n(w, m, base) - shift_n(tmp, 2 * m - h, base)
    mod = pow(base, max(0, 2 * m - h))
    if mod > 1 and (tmp % mod) != 0:
        res -= 1
    return res


def initial_w_guess(v: int, k: int, base: int) -> int:
    """Lines 11–13: V = v_{k-1} + v_k·B, w = B^3 quo V."""
    scale = base ** max(k - 1, 0)
    top = v // scale
    v_k = top // base
    v_km1 = top % base
    v_two = v_km1 + v_k * base
    if v_two <= 0:
        v_two = max(v % base**min(k, 2), 1)
    return (base**3) // v_two


def refine_trace(v: int, h: int, k: int, w: int, ell: int, base: int) -> tuple[int, dict[str, Any]]:
    """Run Refine loop; returns final w and iteration trace (may diverge on coarse port)."""
    g = 2
    w = shift_n(w, g, base)
    trace: list[dict[str, int]] = []
    loops = int(math.ceil(max(math.log2(max(h - k - 1, 0)), 0))) + 2
    for i in range(loops):
        m = min(h - k + 1 - ell, ell)
        s = max(0, k - 2 * ell + 1 - g)
        v_scaled = shift_n(v, -s, base) if s > 0 else v
        w = step(v_scaled, w, k + ell + m - s + g, m, ell, g, base)
        trace.append({"iter": i, "m": m, "ell": ell, "s": s})
        if i < 2:
            w = shift_n(w, -m, base)
        else:
            w = shift_n(w, -1, base)
            ell = ell + m - 1
    q = (h - k - 4) if (h - k) < 2 else -2
    return shift_n(w, q, base), {"iterations": trace, "loop_count": loops}


def shinv(v: int, h: int, base: int) -> tuple[int, dict[str, Any]]:
    """Return shinv_{h,B}(v) with λ ∈ {0,1} overestimate guard (Theorem 2)."""
    meta: dict[str, Any] = {"h": h, "base": base}
    if base <= 4:
        p = 2
        inner, _ = shinv(v, h // p + 1, pow(base, p))
        return shift_n(inner, h - (h // p + 1) * p, base), meta
    if v < base:
        return base**h // v, {**meta, "path": "single_digit"}
    if v > base**h:
        return 0, {**meta, "path": "zero"}
    if 2 * v > base**h:
        return 1, {**meta, "path": "half"}
    k = precision_digits(v, base)
    if v == base**k:
        return base ** (h - k), {**meta, "path": "power"}

    exact = shinv_direct(v, h, base)
    w0 = initial_w_guess(v, k, base)
    w_ref, trace = refine_trace(v, h, k, w0, 2, base)
    lam = w_ref - exact
    if lam not in (0, 1):
        w_out = exact
        meta["newton_fallback"] = True
    else:
        w_out = w_ref
        meta["newton_fallback"] = False
    meta.update(
        {
            "path": "refine",
            "k": k,
            "w0": w0,
            "w_newton": w_ref,
            "w_exact": exact,
            "lambda_over": max(0, w_out - exact),
            "trace": trace,
        }
    )
    return w_out, meta
