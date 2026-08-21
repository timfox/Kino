"""Position encoding (Lehmer code) watermarking for JSON/XML key ordering — Sec. III."""

from __future__ import annotations

import math


def factorial(n: int) -> int:
    return math.factorial(max(0, n))


def min_threshold_T_for_bits(L: int) -> int:
    """Smallest T such that 2^L <= T! (Eq. 12)."""
    if L <= 0:
        return 1
    target = 1 << L
    T = 1
    while factorial(T) < target:
        T += 1
        if T > 10_000:
            raise RuntimeError("T grew unexpectedly large")
    return T


def watermark_bits_to_int(bits: list[int]) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | (1 if b else 0)
    return out


def int_to_watermark_bits(x: int, L: int) -> list[int]:
    x = int(x)
    L = max(0, int(L))
    out = [(x >> (L - 1 - i)) & 1 for i in range(L)] if L else []
    return [int(b) for b in out]


def factorial_decompose(M: int, T: int) -> list[int]:
    """Compute coefficients a_1..a_{T-1} such that M = Σ a_j * (T-j)! (Eq. 2)."""
    T = int(T)
    if T <= 1:
        return []
    M = int(M) % factorial(T)
    coeffs: list[int] = []
    rem = M
    for j in range(1, T):
        base = factorial(T - j)
        a = rem // base
        rem = rem % base
        # Constraints: 0 <= a <= T-j
        a = int(min(max(a, 0), T - j))
        coeffs.append(a)
    # coeffs has length T-1; last implicit coefficient is 0 in Lehmer construction.
    return coeffs


def lehmer_permutation(keys: list[str], coeffs: list[int]) -> list[str]:
    """Apply Lehmer code coefficients to sorted keys (Eq. 4)."""
    T = len(keys)
    if T <= 1:
        return keys[:]
    if len(coeffs) != T - 1:
        raise ValueError("coeffs must have length T-1")
    remain = sorted(keys)
    out: list[str] = []
    for j in range(T - 1):
        a = int(coeffs[j])
        # Valid range 0..len(remain)-1-j; clamp defensively.
        a = min(max(a, 0), len(remain) - 1)
        pick = remain.pop(a)
        out.append(pick)
    out.append(remain[0])
    return out


def invert_lehmer_coeffs(perm: list[str]) -> list[int]:
    """Recover coefficients by reversing the construction (Eq. 6)."""
    T = len(perm)
    if T <= 1:
        return []
    remain = sorted(perm)
    coeffs: list[int] = []
    for j in range(T - 1):
        q = perm[j]
        a = remain.index(q)
        coeffs.append(a)
        remain.pop(a)
    return coeffs


def factorial_recompose(coeffs: list[int], T: int) -> int:
    """Recompose M from coefficients (Eq. 7)."""
    T = int(T)
    if T <= 1:
        return 0
    if len(coeffs) != T - 1:
        raise ValueError("coeffs must have length T-1")
    M = 0
    for j in range(1, T):
        M += int(coeffs[j - 1]) * factorial(T - j)
    return int(M)


def group_keys(keys: list[str], T: int) -> tuple[list[list[str]], list[str]]:
    """Partition keys into full groups of size T; return (groups, leftover)."""
    T = int(T)
    if T <= 0:
        return [], keys[:]
    groups = []
    i = 0
    while i + T <= len(keys):
        groups.append(keys[i : i + T])
        i += T
    return groups, keys[i:]


def embed_watermark_into_keys(keys: list[str], *, watermark_bits: list[int], T: int) -> list[str]:
    """Embed watermark by reordering each full group of T keys; leftover appended unchanged."""
    groups, leftover = group_keys(keys, T)
    L = len(watermark_bits)
    M = watermark_bits_to_int(watermark_bits)
    coeffs = factorial_decompose(M, T)
    out: list[str] = []
    for g in groups:
        out.extend(lehmer_permutation(g, coeffs))
    out.extend(leftover)
    return out


def extract_watermark_from_keys(keys: list[str], *, L: int, T: int) -> list[int]:
    """Extract watermark bits from first complete group (single-group extraction)."""
    groups, _ = group_keys(keys, T)
    if not groups:
        return [0 for _ in range(L)]
    coeffs = invert_lehmer_coeffs(groups[0])
    M = factorial_recompose(coeffs, T)
    return int_to_watermark_bits(M, L)


def majority_vote_bitstrings(bitstrings: list[list[int]]) -> list[int]:
    """Bitwise mode across groups (Eq. 9)."""
    if not bitstrings:
        return []
    L = len(bitstrings[0])
    out: list[int] = []
    for i in range(L):
        ones = sum(1 for b in bitstrings if i < len(b) and b[i] == 1)
        zeros = len(bitstrings) - ones
        out.append(1 if ones >= zeros else 0)
    return out


def extract_with_group_voting(keys: list[str], *, L: int, T: int) -> list[int]:
    """Extract per-group and majority-vote across all full groups."""
    groups, _ = group_keys(keys, T)
    bits = []
    for g in groups:
        coeffs = invert_lehmer_coeffs(g)
        M = factorial_recompose(coeffs, T)
        bits.append(int_to_watermark_bits(M, L))
    return majority_vote_bitstrings(bits)

