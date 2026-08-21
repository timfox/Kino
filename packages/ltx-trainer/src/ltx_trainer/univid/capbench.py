"""CapBench domain catalog and sample cases (arXiv:2606.05748)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapBenchSample:
    sample_id: str
    domain: str
    policy: str
    violative: bool
    ground_truth_caption: str
    ground_truth_events: tuple[str, ...]
    violative_events: tuple[str, ...]


CAPBENCH_DOMAIN_STATS: list[dict[str, int | str]] = [
    {"domain": "Violence", "policy": "Violent Behaviors; Shocking & Graphic Content", "samples": 2700},
    {"domain": "Sex Abuse", "policy": "Exploitation & Abuse; Nudity & Sexual Activity", "samples": 4339},
    {"domain": "Mental Health", "policy": "Mental Health; Harassment & Hateful Behavior", "samples": 1248},
    {"domain": "Regulated Act", "policy": "High-Risk & Regulated Activities", "samples": 4656},
    {"domain": "Integrity", "policy": "Harmful Misinformation; Deceptive Behaviors", "samples": 624},
]

VKB_EXAMPLES: list[dict[str, str]] = [
    {
        "policy": "Nudity & Sexual Activity",
        "event": (
            "The video shows two youths performing a dance that includes sexually suggestive "
            "movements, such as hip thrusting, squatting, and sticking their tongues out."
        ),
    },
    {
        "policy": "High-Risk & Regulated Activities",
        "event": (
            "The video shows multiple people riding on the exterior of moving vehicles, including "
            "on the sides and tops of trucks within a convoy."
        ),
    },
    {
        "policy": "Mental Health",
        "event": (
            "The video contains a real-life clip of a person standing on a train platform as a "
            "train passes, followed by animated scenes depicting bloodstains on the platform."
        ),
    },
    {
        "policy": "Harassment & Hateful Behavior",
        "event": (
            "The video displays a person's full email address and a verification code on a "
            "smartphone screen."
        ),
    },
]


def capbench_catalog() -> dict[str, Any]:
    return {
        "total": 17210,
        "violative": 11476,
        "healthy": 17210 - 11476,
        "domains": CAPBENCH_DOMAIN_STATS,
        "benchmarks_compared": ["Dream-1k", "KuaiMod", "CapBench (ours)"],
    }


def sample_capbench_cases(*, seed: int = 0) -> list[CapBenchSample]:
    cases: list[CapBenchSample] = []
    for i, row in enumerate(CAPBENCH_DOMAIN_STATS):
        domain = str(row["domain"])
        policy = str(row["policy"]).split(";")[0].strip()
        if domain == "Regulated Act":
            events = (
                "The man in the front operates the red motorcycle.",
                "The three men ride together on a single red motorcycle.",
                "None of the three men wear helmets while riding the motorcycle.",
            )
            violative = ("None of the three men wear helmets while riding the motorcycle.",)
            caption = (
                "The video shows three men riding together on a single red motorcycle along a "
                "paved road without helmets. Trees pass by in the background."
            )
        else:
            events = (f"Subject performs action related to {domain.lower()}.",)
            violative = events if i % 2 == 0 else ()
            caption = f"A short-form video clip involving {domain.lower()} content."
        cases.append(
            CapBenchSample(
                sample_id=f"cb-{i:03d}",
                domain=domain,
                policy=policy,
                violative=True,
                ground_truth_caption=caption,
                ground_truth_events=events,
                violative_events=violative,
            )
        )
    return cases
