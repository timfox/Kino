"""Counterfactual text augmentation (Sec. 3.2.2)."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.btscafe.config import BTSCafeConfig

_DEVICE_PATTERNS = (
    re.compile(r"\b(Litt3200|Littmann|AKGC417L|Meditron|Yunting|unknown device)\b", re.I),
    re.compile(r"using a [^.,]+ stethoscope", re.I),
)
_DEMO_PATTERNS = (
    re.compile(r"\badult male patient\b", re.I),
    re.compile(r"\badult female patient\b", re.I),
    re.compile(r"\bpediatric patient\b", re.I),
    re.compile(r"\b(left|right) anterior\b", re.I),
)


def neutralize_device(text: str) -> str:
    out = text
    out = _DEVICE_PATTERNS[1].sub("using an unknown device", out)
    out = _DEVICE_PATTERNS[0].sub("unknown device", out)
    return out


def counterfactual_text_augment(
    text: str,
    *,
    cfg: BTSCafeConfig | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Neutralize device metadata always; other demographics with probability p_text.
    """
    import random

    cfg = cfg or BTSCafeConfig()
    rng = random.Random(seed)

    original = text
    augmented = neutralize_device(text)
    device_neutralized = augmented != original

    demo_neutralized = False
    if rng.random() < cfg.p_text_neutralize:
        demo_text = augmented
        for pat in _DEMO_PATTERNS:
            demo_text = pat.sub("an adult patient of unknown sex", demo_text)
        demo_neutralized = demo_text != augmented
        augmented = demo_text

    return {
        "original": original,
        "augmented": augmented,
        "device_neutralized": device_neutralized,
        "demographic_neutralized": demo_neutralized,
    }
