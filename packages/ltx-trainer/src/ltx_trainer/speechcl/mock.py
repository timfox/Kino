"""Speech CL survey smoke — geometry drift vs adaptation modes (arXiv:2605.24863)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def compare_adaptation_modes(*, seed: int = 42) -> dict[str, Any]:
    import numpy as np

    geom = load_sibling(__file__, "geometry")
    rng = np.random.default_rng(seed)
    emb, labels = geom.simulate_entangled_embeddings(24, rng)
    sep0 = geom._class_separability(emb, labels)

    emb_full = geom.apply_adaptation_drift(emb, 1.0, rng)
    sep_full = geom._class_separability(emb_full, labels)
    full_drop = (sep0 - sep_full) / max(sep0, 1e-9)

    emb_lora = geom.peft_style_update(emb, 0.6, rng)
    sep_lora = geom._class_separability(emb_lora, labels)
    lora_drop = (sep0 - sep_lora) / max(sep0, 1e-9)

    return {
        "lora_relative_drop": round(float(lora_drop), 4),
        "full_update_relative_drop": round(float(full_drop), 4),
        "full_update_hurts_more": bool(full_drop > lora_drop),
        "separability_before": round(float(sep0), 4),
    }


def evaluation_smoke(*, seed: int = 42) -> dict[str, Any]:
    cfg = load_sibling(__file__, "config").SpeechClConfig()
    geom = load_sibling(__file__, "geometry")
    drift = geom.drift_report(seed=seed, drift_strength=0.8)
    adapt = compare_adaptation_modes(seed=seed)
    return {
        "paper": cfg.paper_arxiv,
        **adapt,
        "phonetic_separability_before": round(drift["phonetic_separability_before"], 4),
        "phonetic_separability_after": round(drift["phonetic_separability_after"], 4),
        "relative_drop": round(drift["relative_drop"], 4),
    }
