"""FAST-ME stopping smoke."""

from __future__ import annotations

from typing import Any


def toy_sad_sequence() -> list[float]:
    return [5000.0, 4200.0, 3800.0, 3500.0]


def toy_attention_high_motion() -> float:
    return 0.85


def toy_attention_background() -> float:
    return 0.12


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.fast_me.blend import blended_cost, should_stop_fast_me

    sad = toy_sad_sequence()[-1]
    attn = toy_attention_high_motion()
    cost = blended_cost(sad, attn, alpha=0.6)
    stop = should_stop_fast_me(sad, attn, best_sad=4000.0)
    return {"blended_cost": round(cost, 2), "should_stop": stop, "sad_final": sad}
