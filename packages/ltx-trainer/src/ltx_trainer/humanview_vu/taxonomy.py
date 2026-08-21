"""Watch–Remember–Reason taxonomy (Sec. 3, Fig. 1)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class CoreAbility(str, Enum):
    WATCH = "watch"
    REMEMBER = "remember"
    REASON = "reason"


class WatchBranch(str, Enum):
    FINE_GRAINED = "fine_grained_watching"
    COMPREHENSIVE = "comprehensive_watching"
    AUDIO_VISUAL = "audio_visual_watching"
    EFFICIENT = "efficient_watching"


class RememberBranch(str, Enum):
    OFFLINE_AGENTIC = "offline_memory_agentic"
    OFFLINE_NON_AGENT = "offline_memory_non_agent"
    STREAMING = "streaming_memory"


class ReasonBranch(str, Enum):
    TEXT_AGENTIC = "text_only_reasoning_agentic"
    TEXT_NON_AGENT = "text_only_reasoning_non_agent"
    THINK_VID_AGENTIC = "thinking_with_videos_agentic"
    THINK_VID_NON_AGENT = "thinking_with_videos_non_agent"


WATCH_METHODS: dict[WatchBranch, list[str]] = {
    WatchBranch.FINE_GRAINED: [
        "TimeChat",
        "TRACE",
        "Sa2VA",
        "SAMA",
        "TimeLens",
        "UniTime",
        "OMTG",
    ],
    WatchBranch.COMPREHENSIVE: [
        "Streaming-DVC",
        "AuroraCap",
        "Tarsier2",
        "AnyCap",
        "DoYouRemember",
    ],
    WatchBranch.AUDIO_VISUAL: [
        "Qwen3-Omni",
        "OmniVinci",
        "Ming-Omni",
        "LLaMA-Omni",
        "Qwen2.5-Omni",
    ],
    WatchBranch.EFFICIENT: [
        "AKS",
        "Q-Frame",
        "FrameFusion",
        "Video-XL-2",
        "VideoNSA",
        "DyCoke",
    ],
}

REMEMBER_METHODS: dict[RememberBranch, list[str]] = {
    RememberBranch.OFFLINE_AGENTIC: [
        "VideoAgent",
        "LVAgent",
        "VideoLucy",
        "AdaVideoRAG",
        "M3-Agent",
        "MemGen",
        "GCAgent",
        "EGAgent",
    ],
    RememberBranch.OFFLINE_NON_AGENT: [
        "MovieChat",
        "MA-LMM",
        "ReWind",
        "LongVU",
        "VidCompress",
        "MARC",
        "VideoLLaMB",
        "HERMES",
        "∞-Video",
    ],
    RememberBranch.STREAMING: [
        "Flash-VStream",
        "StreamMem",
        "InfiniPot-V",
        "StreamChat",
        "StreamingVLM",
        "ReKV",
        "ProVideLLM",
    ],
}

REASON_METHODS: dict[ReasonBranch, list[str]] = {
    ReasonBranch.TEXT_AGENTIC: [
        "VideoAgent",
        "DoraemonGPT",
        "Video-of-Thought",
        "VCA",
        "Flow4Agent",
        "VideoAgent2",
        "CoT-Vid",
        "DVD",
    ],
    ReasonBranch.TEXT_NON_AGENT: [
        "Video-R1",
        "VideoRFT",
        "VerIPO",
        "DeepVideo-R1",
        "TW-GRPO",
        "VistaDPO",
        "Time-R1",
        "Video-CoT",
    ],
    ReasonBranch.THINK_VID_AGENTIC: [
        "Open-o3-Video",
        "Video-Thinker",
        "Rewatch-R1",
        "VideoChat-R1.5",
        "VITAL",
        "Conan",
        "FrameMind",
        "Video-o3",
        "Pixel Reasoner",
        "VideoZoomer",
    ],
    ReasonBranch.THINK_VID_NON_AGENT: [
        "Open-o3-Video",
        "Video-Thinker",
        "Rewatch-R1",
    ],
}

_METHOD_INDEX: dict[str, tuple[CoreAbility, str]] = {}


def _build_index() -> None:
    if _METHOD_INDEX:
        return
    for branch, names in WATCH_METHODS.items():
        for name in names:
            _METHOD_INDEX[name.lower()] = (CoreAbility.WATCH, branch.value)
    for branch, names in REMEMBER_METHODS.items():
        for name in names:
            _METHOD_INDEX[name.lower()] = (CoreAbility.REMEMBER, branch.value)
    for branch, names in REASON_METHODS.items():
        for name in names:
            _METHOD_INDEX[name.lower()] = (CoreAbility.REASON, branch.value)


def classify_method(name: str) -> dict[str, str | None]:
    """Map a paper method name to core ability + leaf branch."""
    _build_index()
    key = name.strip().lower().replace("_", "-")
    for alias, (ability, branch) in _METHOD_INDEX.items():
        if alias in key or key in alias:
            return {"method": name, "ability": ability.value, "branch": branch}
    if any(tok in key for tok in ("stream", "kv", "memory", "rag", "mem")):
        return {"method": name, "ability": CoreAbility.REMEMBER.value, "branch": None}
    if any(tok in key for tok in ("r1", "grpo", "cot", "reason", "o3", "think")):
        return {"method": name, "ability": CoreAbility.REASON.value, "branch": None}
    if any(tok in key for tok in ("omni", "audio", "caption", "ground", "frame", "vtg")):
        return {"method": name, "ability": CoreAbility.WATCH.value, "branch": None}
    return {"method": name, "ability": None, "branch": None}


def taxonomy_card() -> dict[str, Any]:
    return {
        "core_abilities": [a.value for a in CoreAbility],
        "watch": {b.value: WATCH_METHODS[b] for b in WatchBranch},
        "remember": {b.value: REMEMBER_METHODS[b] for b in RememberBranch},
        "reason": {b.value: REASON_METHODS[b] for b in ReasonBranch},
    }
