"""Top-K peer similarity for augmented POI pool (Sec. III-D1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from typing import Sequence

from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.profiles import IndividualProfile, build_profile_from_trajectories
from ltx_trainer.trajgenagent.trajectory import DailyTrajectory


@dataclass(frozen=True)
class PeerMatch:
    peer_id: str
    similarity: float


def _activity_vector(profile: IndividualProfile, vocabulary: tuple[str, ...]) -> list[float]:
    return [profile.activity_freq.get(a, 0.0) for a in vocabulary]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def rank_peers(
    target: IndividualProfile,
    candidates: dict[str, IndividualProfile],
    *,
    vocabulary: tuple[str, ...] | None = None,
    top_k: int = 5,
) -> tuple[PeerMatch, ...]:
    vocab = vocabulary or tuple(sorted(set(target.activity_freq) | {*sum((list(c.activity_freq) for c in candidates.values()), [])}))
    target_vec = _activity_vector(target, vocab)
    scored: list[PeerMatch] = []
    for peer_id, profile in candidates.items():
        if peer_id == target.individual_id:
            continue
        sim = cosine_similarity(target_vec, _activity_vector(profile, vocab))
        scored.append(PeerMatch(peer_id=peer_id, similarity=sim))
    scored.sort(key=lambda m: -m.similarity)
    return tuple(scored[:top_k])


def augmented_poi_pool(
    activity: str,
    target: IndividualProfile,
    peers: Sequence[IndividualProfile],
) -> dict[str, float]:
    """P_u(a) ∪ P_sim(a) frequency map for spatial worker."""
    pool: dict[str, float] = dict(target.poi_given_activity.get(activity, {}))
    for peer in peers:
        for poi_id, freq in peer.poi_given_activity.get(activity, {}).items():
            pool[poi_id] = pool.get(poi_id, 0.0) + freq * 0.5
    total = sum(pool.values()) or 1.0
    return {k: v / total for k, v in pool.items()}


def peer_retrieval_demo(
    trajectories_by_user: dict[str, tuple[DailyTrajectory, ...]],
    *,
    target_id: str = "u_001",
    cfg: TrajGenAgentConfig | None = None,
) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    profiles = {
        uid: build_profile_from_trajectories(uid, trajs) for uid, trajs in trajectories_by_user.items()
    }
    target = profiles[target_id]
    peers = rank_peers(target, profiles, top_k=cfg.peer_top_k)
    peer_profiles = [profiles[m.peer_id] for m in peers]
    work_pois = augmented_poi_pool("Work", target, peer_profiles)
    return {
        "target": target_id,
        "top_peers": [m.__dict__ for m in peers],
        "augmented_work_poi_count": len(work_pois),
        "sample_work_pois": list(work_pois.items())[:3],
    }
