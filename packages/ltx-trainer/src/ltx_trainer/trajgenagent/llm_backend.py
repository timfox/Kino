"""Optional vLLM backend with deterministic profile fallback."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from ltx_trainer.trajgenagent.config import ACTIVITY_VOCABULARY, TrajGenAgentConfig
from ltx_trainer.trajgenagent.profiles import IndividualProfile


def vllm_base_url(cfg: TrajGenAgentConfig | None = None) -> str:
    cfg = cfg or TrajGenAgentConfig()
    return os.environ.get("GOPEX_TRAJGENAGENT_VLLM_URL", cfg.vllm_base_url).rstrip("/")


def vllm_available(cfg: TrajGenAgentConfig | None = None) -> bool:
    if os.environ.get("GOPEX_TRAJGENAGENT_FORCE_STUB", "").strip() in ("1", "true", "yes"):
        return False
    url = f"{vllm_base_url(cfg)}/models"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _chat_completion(prompt: str, *, cfg: TrajGenAgentConfig) -> str | None:
    payload = {
        "model": cfg.backbone_llm,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.llm_temperature,
        "top_p": cfg.llm_top_p,
        "max_tokens": cfg.llm_max_tokens,
    }
    req = urllib.request.Request(
        f"{vllm_base_url(cfg)}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.vllm_timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError):
        return None


def _parse_activity_chain(text: str, vocabulary: tuple[str, ...] = ACTIVITY_VOCABULARY) -> tuple[str, ...] | None:
    vocab = set(vocabulary)
    found = re.findall(r"\b(" + "|".join(re.escape(v) for v in vocabulary) + r")\b", text)
    chain = tuple(a for a in found if a in vocab)
    return chain if chain else None


def orchestrator_prompt(
    *,
    weekday: str,
    day_type: str,
    profile: IndividualProfile,
    vocabulary: tuple[str, ...] = ACTIVITY_VOCABULARY,
) -> str:
    exemplars = "\n".join(" → ".join(c) for c in profile.exemplar_chains[:3])
    top_acts = ", ".join(f"{a}:{p:.2f}" for a, p in sorted(profile.activity_freq.items(), key=lambda x: -x[1])[:6])
    return (
        "Generate one daily activity chain as arrow-separated activities.\n"
        f"Weekday: {weekday}; day_type: {day_type}.\n"
        f"Allowed activities: {', '.join(vocabulary)}.\n"
        "Rules: start and end at Home; no adjacent duplicates.\n"
        f"User activity mix: {top_acts}.\n"
        f"Exemplars:\n{exemplars}\n"
        "Reply with one chain only, e.g. Home → Work → EatOut → Work → Home"
    )


def generate_activity_chain_llm(
    *,
    weekday: str,
    day_type: str,
    profile: IndividualProfile,
    cfg: TrajGenAgentConfig | None = None,
    vocabulary: tuple[str, ...] = ACTIVITY_VOCABULARY,
) -> tuple[str, ...] | None:
    cfg = cfg or TrajGenAgentConfig()
    if not cfg.use_llm or not vllm_available(cfg):
        return None
    text = _chat_completion(orchestrator_prompt(weekday=weekday, day_type=day_type, profile=profile), cfg=cfg)
    if not text:
        return None
    return _parse_activity_chain(text, vocabulary)


def duration_prompt(
    activity: str,
    *,
    remaining: tuple[str, ...],
    profile: IndividualProfile,
) -> str:
    mean = profile.duration_mean_min.get(activity, 60.0)
    return (
        f"Estimate stay duration in minutes for activity '{activity}'.\n"
        f"Historical mean: {mean:.0f} min.\n"
        f"Remaining activities: {', '.join(remaining) if remaining else '(none)'}.\n"
        "Reply JSON: {\"duration_minutes\": number}"
    )


def parse_duration_json(text: str) -> float | None:
    try:
        obj = json.loads(text.strip())
        return float(obj["duration_minutes"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        match = re.search(r"duration_minutes[\"']?\s*[:=]\s*([0-9.]+)", text)
        return float(match.group(1)) if match else None


def estimate_duration_llm(
    activity: str,
    *,
    remaining_activities: tuple[str, ...],
    profile: IndividualProfile,
    cfg: TrajGenAgentConfig | None = None,
) -> float | None:
    cfg = cfg or TrajGenAgentConfig()
    if not cfg.use_llm or not vllm_available(cfg):
        return None
    text = _chat_completion(
        duration_prompt(activity, remaining=remaining_activities, profile=profile),
        cfg=cfg,
    )
    return parse_duration_json(text) if text else None


def backend_status(cfg: TrajGenAgentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TrajGenAgentConfig()
    force_stub = os.environ.get("GOPEX_TRAJGENAGENT_FORCE_STUB", "").strip().lower() in ("1", "true", "yes")
    live = vllm_available(cfg)
    return {
        "use_llm": cfg.use_llm,
        "vllm_base_url": vllm_base_url(cfg),
        "vllm_live": live,
        "available": live and cfg.use_llm,
        "force_stub": force_stub,
        "backbone": cfg.backbone_llm,
        "mode": "vllm" if live and cfg.use_llm else "profile_deterministic",
    }
