"""Routed ground-truth rewards for GRPO (Sec. 3.2.2, Eq. 5)."""

from __future__ import annotations

import re
from typing import Literal

from ltx_trainer.panoenv.config import REWARD_W_ACC, REWARD_W_FMT

RewardStrategy = Literal["yes_no", "mcq", "distance", "spatial", "counting"]

_ANSWER_RE = re.compile(r"<Answer>\s*(.*?)\s*</Answer>", re.I | re.S)
_FMT_RE = re.compile(
    r"<Reasoning>.*?</Reasoning>\s*<Answer>.*?</Answer>",
    re.I | re.S,
)

_SPATIAL_SYNONYMS: dict[str, set[str]] = {
    "above": {"above", "over", "on top"},
    "below": {"below", "under", "beneath"},
    "front": {"front", "ahead", "forward", "in front"},
    "back": {"back", "behind", "rear"},
    "left": {"left"},
    "right": {"right"},
}


def extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def format_reward(response: str) -> float:
    return 1.0 if _FMT_RE.search(response) else 0.0


def strategy_yes_no(pred: str, gt: str) -> float:
    return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0


def _normalize_entity(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", "", s.lower())
    return re.sub(r"\b(a|an|the)\b", "", s).strip()


def strategy_mcq(pred: str, gt: str) -> float:
    p = _normalize_entity(pred.split()[0] if len(pred.split()) > 3 else pred)
    g = _normalize_entity(gt.split()[0] if len(gt.split()) > 3 else gt)
    return 1.0 if p == g or p in g or g in p else 0.0


def _parse_meters(text: str) -> float | None:
    m = re.search(r"([\d.]+)\s*(m|meter|meters|cm|km|ft)?", text.lower())
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "m").lower()
    if unit == "cm":
        return val / 100.0
    if unit == "km":
        return val * 1000.0
    if unit == "ft":
        return val * 0.3048
    return val


def strategy_distance(pred: str, gt: str) -> float:
    p, g = _parse_meters(pred), _parse_meters(gt)
    if p is None or g is None or g == 0:
        return 0.0
    rel = abs(p - g) / abs(g)
    if rel <= 0.10:
        return 1.0
    if rel <= 0.20:
        return 0.5
    return 0.0


def _spatial_axes(text: str) -> dict[str, set[str]]:
    t = text.lower()
    found: dict[str, set[str]] = {}
    for axis, words in _SPATIAL_SYNONYMS.items():
        if any(w in t for w in words):
            found[axis] = words
    return found


def strategy_spatial(pred: str, gt: str) -> float:
    p_ax = _spatial_axes(pred)
    g_ax = _spatial_axes(gt)
    if not g_ax:
        return 0.0
    matched = sum(1 for k in g_ax if k in p_ax)
    return matched / len(g_ax)


def strategy_counting(pred: str, gt: str) -> float:
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    def num(s: str) -> int | None:
        m = re.search(r"\d+", s)
        if m:
            return int(m.group())
        for w, n in word_map.items():
            if w in s.lower():
                return n
        return None

    p, g = num(pred), num(gt)
    return 1.0 if p is not None and p == g else 0.0


def routed_accuracy(pred: str, gt: str, strategy: RewardStrategy) -> float:
    ans = extract_answer(pred)
    if strategy == "yes_no":
        return strategy_yes_no(ans, gt)
    if strategy == "mcq":
        return strategy_mcq(ans, gt)
    if strategy == "distance":
        return strategy_distance(ans, gt)
    if strategy == "spatial":
        return strategy_spatial(ans, gt)
    if strategy == "counting":
        return strategy_counting(ans, gt)
    return 0.0


def total_reward(
    response: str,
    gt: str,
    strategy: RewardStrategy,
    *,
    w_acc: float = REWARD_W_ACC,
    w_fmt: float = REWARD_W_FMT,
) -> float:
    r_acc = routed_accuracy(response, gt, strategy)
    r_fmt = format_reward(response)
    return w_acc * r_acc + w_fmt * r_fmt
