"""Quotient + remainder via shifted inverse — Algorithm 3 (Theorem 2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.midint_division.digits import precision_digits, shinv_direct, shift_n
from ltx_trainer.midint_division.shifted_inverse import shinv


def divide(
    u: int,
    v: int,
    *,
    base: int = 10,
    use_newton_shinv: bool = True,
) -> tuple[int, int, dict[str, Any]]:
    """Div(u, v) with δ ∈ {-1, 0, 1} quotient correction (revised Theorem 2)."""
    if v <= 0:
        raise ValueError("divisor must be positive")
    if u < 0:
        raise ValueError("unsigned dividend required")
    h = precision_digits(u, base)
    meta: dict[str, Any] = {"h": h, "base": base}
    if use_newton_shinv:
        w, sh_meta = shinv(v, h, base)
        meta["shinv"] = sh_meta
    else:
        w = shinv_direct(v, h, base)
    exact_w = shinv_direct(v, h, base)
    meta["shinv_lambda"] = w - exact_w
    q0 = shift_n(u * w, -h, base)
    m = q0 * v
    delta = 0
    if u < m:
        q = q0 - 1
        delta = -1
        r = u - q * v
    else:
        q = q0
        r = u - m
    if r >= v:
        q += 1
        delta = 1
        r -= v
    meta["q0"] = q0
    meta["delta"] = delta
    return q, r, meta


def verify_example(u: int, v: int, expected_q: int, *, base: int = 10) -> dict[str, Any]:
    q, r, meta = divide(u, v, base=base)
    ok = q == expected_q and 0 <= r < v and q * v + r == u
    return {"u": u, "v": v, "q": q, "r": r, "expected_q": expected_q, "ok": ok, **meta}
