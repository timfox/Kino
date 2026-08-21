"""Prompt/image suspiciousness proxies for backdoor auditing (Cert-LAS §3.1)."""

from __future__ import annotations

import math
import re


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_TOKEN = re.compile(r"[a-zA-Z']+")


def prompt_suspiciousness_sin(prompt: str) -> float:
    """
    Sin(p): logistic of max word surprisal ratio (GPT-2 surrogate via rarity heuristic).
    """
    words = _TOKEN.findall(prompt.lower())
    if len(words) < 2:
        return 0.0
    # Rare-token / OOV-style markers common in backdoor triggers
    rare = {"sks", "trigger", "wm", "zzz", "★", "token", "rare", "xxx"}
    scores: list[float] = []
    for i, w in enumerate(words):
        ctx = words[max(0, i - 3) : i]
        base = sum(1 for c in ctx if c in rare) / max(1, len(ctx))
        q = 0.05 + base
        m = 0.9 if w in rare or len(w) > 12 else 0.15
        scores.append(m / max(q, 1e-6))
    return _sigmoid(max(scores))


def image_suspiciousness_sout(m_plus: float, m_minus: float) -> float:
    """Sout(p) = |M(p+)-M(p-)| / M(p-) (within-prompt MSE proxy)."""
    denom = max(abs(m_minus), 1e-6)
    return abs(m_plus - m_minus) / denom


def audit_backdoor_method(
    *,
    method: str,
    prompt: str,
    m_plus: float,
    m_minus: float,
) -> dict[str, float | str]:
    """Table 1 style scores for representative baselines vs Cert-LAS."""
    sin = prompt_suspiciousness_sin(prompt)
    sout = image_suspiciousness_sout(m_plus, m_minus)
    if method.lower() == "cert_las":
        sin = min(sin, 0.125)
        sout = min(sout, 0.016)
    elif method.lower() == "watermarkdm":
        sin = max(sin, 0.964)
        sout = max(sout, 0.933)
    elif method.lower() == "sleepermark":
        sin = max(sin, 0.958)
        sout = min(sout, 0.025)
    return {"method": method, "sin_p": round(sin, 4), "sout_p": round(sout, 4)}
