"""Individual evidence selection + peer context (Sec. III-C1, III-D1)."""

from __future__ import annotations

from typing import Sequence

from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.corpus import build_synthetic_corpus
from ltx_trainer.trajgenagent.peer_retrieval import peer_retrieval_demo, rank_peers
from ltx_trainer.trajgenagent.profiles import (
    IndividualProfile,
    build_profile_from_trajectories,
    profile_summary,
)
from ltx_trainer.trajgenagent.trajectory import DailyTrajectory


def select_evidence_days(
    trajectories: Sequence[DailyTrajectory],
    *,
    weekday: str,
    day_type: str = "weekday",
) -> tuple[DailyTrajectory, ...]:
    same_day = [t for t in trajectories if t.weekday == weekday]
    if same_day:
        return tuple(same_day[:5])
    if day_type == "weekday":
        pool = [t for t in trajectories if t.weekday not in ("Saturday", "Sunday")]
    else:
        pool = [t for t in trajectories if t.weekday in ("Saturday", "Sunday")]
    if pool:
        return tuple(pool[:5])
    return tuple(trajectories[:5])


def build_target_context(
    trajectories_by_user: dict[str, tuple[DailyTrajectory, ...]],
    *,
    target_id: str = "u_001",
    cfg: TrajGenAgentConfig | None = None,
) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    profiles = {
        uid: build_profile_from_trajectories(uid, trajs) for uid, trajs in trajectories_by_user.items()
    }
    if target_id not in profiles:
        target_id = next(iter(profiles))
    target = profiles[target_id]
    peers = rank_peers(target, profiles, top_k=cfg.peer_top_k)
    peer_profiles = tuple(profiles[m.peer_id] for m in peers)
    return {
        "profile": target,
        "peer_profiles": peer_profiles,
        "peer_matches": peers,
    }


def evidence_demo(*, seed: int = 42) -> dict[str, object]:
    users = build_synthetic_corpus(seed=seed)
    target_trajs = users["u_001"]
    profile = build_profile_from_trajectories("u_001", target_trajs)
    evidence = select_evidence_days(target_trajs, weekday="Monday")
    peers = peer_retrieval_demo(users, target_id="u_001")
    return {
        "profile": profile_summary(profile),
        "evidence_days": len(evidence),
        "peer_retrieval": peers,
    }
