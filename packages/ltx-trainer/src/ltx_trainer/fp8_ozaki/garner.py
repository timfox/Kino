"""Garner CRT reconstruction (Ozaki II Phase 3, Eq. 7 / Appendix A)."""

from __future__ import annotations


def mod_inverse(a: int, m: int) -> int:
    """Extended Euclidean algorithm for a^{-1} mod m."""
    if m <= 1:
        raise ValueError("modulus must be > 1")
    t, new_t = 0, 1
    r, new_r = m, a % m
    while new_r != 0:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r
    if r != 1:
        raise ValueError(f"{a} has no inverse mod {m}")
    return t % m


def garner_reconstruct(residues: list[int], moduli: list[int]) -> int:
    """Mixed-radix CRT digits via Garner's algorithm (Eq. 7)."""
    if len(residues) != len(moduli):
        raise ValueError("residues and moduli length mismatch")
    if not moduli:
        return 0
    digits: list[int] = []
    for k in range(len(moduli)):
        mk = moduli[k]
        partial = residues[k]
        prod = 1
        for j in range(k):
            partial = (partial - digits[j] * prod) % mk
            prod *= moduli[j]
        inv = mod_inverse(prod % mk, mk)
        digits.append((partial * inv) % mk)
    result = 0
    base = 1
    for d, m in zip(digits, moduli, strict=True):
        result += d * base
        base *= m
    return result


def precompute_garner_inverses(moduli: list[int]) -> list[int]:
    """Prefix products and modular inverses for constant-memory reconstruction."""
    inverses: list[int] = []
    prod = 1
    for k in range(1, len(moduli)):
        prod = (prod * moduli[k - 1]) % moduli[k]
        inverses.append(mod_inverse(prod, moduli[k]))
    return inverses
