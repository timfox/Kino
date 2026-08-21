"""Score a rendered frame against a FreeUSD shot graph (CPU, no GPU)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.freeusd.usda import parse_character_names, parse_expected_cast, parse_lighting


LOOK_BLEED = {
    "bw": "look_bleed_bw",
    "noir": "look_bleed_noir",
    "daytime": None,
    "unknown": "look_unknown",
}


def score_frame_vs_usd(frame: dict[str, Any], usda: str) -> dict[str, Any]:
    """Compare frame_fail.assess() output to USDA expected cast/lighting/blocking."""
    expected_n = parse_expected_cast(usda)
    lighting = parse_lighting(usda)
    names = parse_character_names(usda)
    look = str(frame.get("look") or "unknown")
    count = frame.get("person_count")
    count_ok = frame.get("person_count_ok")
    failures: list[str] = []
    if lighting == "high-key" and not frame.get("look_ok"):
        tag = LOOK_BLEED.get(look) or "look_bleed"
        failures.append(tag)
    if count_ok and isinstance(count, int) and count != expected_n:
        failures.append("person_count" if count < expected_n else "extras")
    if count_ok and isinstance(count, int) and count > expected_n:
        if "extras" not in failures:
            failures.append("extras")
    if frame.get("shot_size_guess") == "MCU" and "MS" in str(frame.get("expected_shot_size") or "MS"):
        failures.append("spatial_blocking")
    soap_ok = not failures and bool(frame.get("look_ok"))
    if count_ok is False and soap_ok:
        # Lighting passed but we could not count people — do not promote as identity lock.
        soap_ok = False
        failures.append("identity_unverified")
    return {
        "expected_cast": expected_n,
        "expected_names": names,
        "expected_lighting": lighting,
        "observed_look": look,
        "observed_person_count": count,
        "failures": failures,
        "soap_ok": soap_ok,
        "spatial_lock": True,
    }
