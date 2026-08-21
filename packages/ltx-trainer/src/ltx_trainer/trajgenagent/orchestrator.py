"""Stage 1 orchestrator: activity-chain ICL + verify loop (Sec. III-C)."""

from __future__ import annotations

import random
from typing import Sequence

from ltx_trainer.trajgenagent.config import ACTIVITY_VOCABULARY, TrajGenAgentConfig
from ltx_trainer.trajgenagent.llm_backend import generate_activity_chain_llm
from ltx_trainer.trajgenagent.profiles import IndividualProfile, build_profile_from_activity_chains


def validate_activity_chain(
    chain: Sequence[str],
    *,
    vocabulary: Sequence[str] = ACTIVITY_VOCABULARY,
) -> tuple[bool, str]:
    if not chain:
        return False, "empty_chain"
    vocab = set(vocabulary)
    for a in chain:
        if a not in vocab:
            return False, "vocabulary"
    for i in range(1, len(chain)):
        if chain[i] == chain[i - 1]:
            return False, "adjacent_duplicate"
    if chain[0] != "Home" or chain[-1] != "Home":
        return False, "home_anchor"
    return True, "ok"


def synthesize_activity_chain_from_profile(
    *,
    weekday: str = "Monday",
    day_type: str = "weekday",
    profile: IndividualProfile | None = None,
    rng: random.Random | None = None,
    vocabulary: Sequence[str] = ACTIVITY_VOCABULARY,
    cfg: TrajGenAgentConfig | None = None,
) -> tuple[str, ...]:
    """Sample a valid chain from Πu transition frequencies (ICL stand-in)."""
    cfg = cfg or TrajGenAgentConfig()
    rng = rng or random.Random(hash((weekday, day_type, profile.individual_id if profile else "")) % 2**32)
    profile = profile or build_profile_from_activity_chains(
        "default",
        [
            ("Home", "Work", "EatOut", "Work", "Home"),
            ("Home", "Work", "Shop", "Work", "Home"),
        ],
    )
    if day_type == "weekend" or weekday in ("Saturday", "Sunday"):
        base = ("Home", "Leisure", "EatOut", "Home")
        return base if validate_activity_chain(base, vocabulary=vocabulary)[0] else ("Home", "Leisure", "Home")

    chain: list[str] = ["Home"]
    max_steps = 12
    while chain[-1] != "Home" or len(chain) == 1:
        if len(chain) >= max_steps:
            chain.append("Home")
            break
        prev = chain[-1]
        candidates = [
            (k.split("->", 1)[1], p)
            for k, p in profile.transition_freq.items()
            if k.startswith(f"{prev}->")
        ]
        if not candidates and prev != "Home":
            chain.append("Home")
            break
        if not candidates:
            candidates = [("Work", 0.5), ("Leisure", 0.2), ("Home", 0.3)]
        acts, weights = zip(*candidates, strict=True)
        nxt = rng.choices(list(acts), weights=list(weights), k=1)[0]
        if nxt == prev:
            continue
        chain.append(nxt)
        if nxt == "Home" and len(chain) > 2:
            break
    if chain[-1] != "Home":
        chain.append("Home")
    ok, _ = validate_activity_chain(chain, vocabulary=vocabulary)
    if ok:
        return tuple(chain)
    return ("Home", "Work", "EatOut", "Work", "Home")


def synthesize_activity_chain_stub(
    *,
    weekday: str = "Monday",
    day_type: str = "weekday",
    profile: IndividualProfile | None = None,
    cfg: TrajGenAgentConfig | None = None,
) -> tuple[str, ...]:
    return synthesize_activity_chain_from_profile(
        weekday=weekday,
        day_type=day_type,
        profile=profile,
        cfg=cfg,
    )


def generate_activity_chain(
    *,
    weekday: str = "Monday",
    day_type: str = "weekday",
    historical_chains: Sequence[Sequence[str]] | None = None,
    profile: IndividualProfile | None = None,
    cfg: TrajGenAgentConfig | None = None,
    vocabulary: Sequence[str] = ACTIVITY_VOCABULARY,
) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    chains = historical_chains or profile.exemplar_chains if profile else [
        ("Home", "Work", "EatOut", "Work", "Home"),
        ("Home", "Work", "Shop", "Work", "Home"),
    ]
    prof = profile or build_profile_from_activity_chains("u_local", chains)
    source = "profile_sampler"
    for attempt in range(cfg.orchestrator_max_retries + 1):
        llm_chain = generate_activity_chain_llm(
            weekday=weekday,
            day_type=day_type,
            profile=prof,
            cfg=cfg,
            vocabulary=tuple(vocabulary),
        )
        if llm_chain is not None:
            ok, reason = validate_activity_chain(llm_chain, vocabulary=vocabulary)
            if ok:
                return {
                    "chain": llm_chain,
                    "valid": True,
                    "attempts": attempt + 1,
                    "source": "vllm",
                    "evidence_exemplars": prof.exemplar_chains,
                    "activity_freq": prof.activity_freq,
                }
            source = f"vllm_invalid:{reason}"
        chain = synthesize_activity_chain_from_profile(
            weekday=weekday,
            day_type=day_type,
            profile=prof,
            vocabulary=vocabulary,
            cfg=cfg,
        )
        ok, reason = validate_activity_chain(chain, vocabulary=vocabulary)
        if ok:
            return {
                "chain": chain,
                "valid": True,
                "attempts": attempt + 1,
                "source": source,
                "evidence_exemplars": prof.exemplar_chains,
                "activity_freq": prof.activity_freq,
            }
    fallback = tuple(chains[0])
    return {
        "chain": fallback,
        "valid": True,
        "attempts": cfg.orchestrator_max_retries + 1,
        "fallback": True,
        "source": "fallback",
    }
