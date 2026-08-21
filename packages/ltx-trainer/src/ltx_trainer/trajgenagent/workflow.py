"""Deterministic LangGraph-style worker workflow (Sec. III-D, Eq. 5)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Sequence

from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.corpus import profile_for_individual, reference_corpus_for_dataset
from ltx_trainer.trajgenagent.evidence import build_target_context
from ltx_trainer.trajgenagent.orchestrator import generate_activity_chain
from ltx_trainer.trajgenagent.profiles import IndividualProfile
from ltx_trainer.trajgenagent.spatial_worker import sample_poi
from ltx_trainer.trajgenagent.temporal_worker import estimate_duration, estimate_travel_minutes, cold_start_time
from ltx_trainer.trajgenagent.trajectory import DailyTrajectory, Visit
from ltx_trainer.trajgenagent.workflow_graph import verify_visit_bounds


@dataclass
class WorkflowState:
    individual_id: str
    date: str
    weekday: str
    day_type: str
    activity_chain: tuple[str, ...]
    visits: list[Visit] = field(default_factory=list)
    visit_index: int = 0
    current_time: object | None = None
    prev_lat: float | None = None
    prev_lon: float | None = None
    prev_activity: str | None = None
    profile: IndividualProfile | None = None
    peer_profiles: Sequence[IndividualProfile] = field(default_factory=tuple)
    tool_calls_completed: int = 0
    tool_calls_required: int = 0


def _ground_visit(state: WorkflowState, *, rng: random.Random, cfg: TrajGenAgentConfig) -> bool:
    i = state.visit_index
    if i >= len(state.activity_chain):
        return False
    activity = state.activity_chain[i]
    poi = sample_poi(
        activity,
        prev_lat=state.prev_lat,
        prev_lon=state.prev_lon,
        prev_activity=state.prev_activity,
        profile=state.profile,
        peer_profiles=state.peer_profiles,
        rng=rng,
        cfg=cfg,
    )
    if poi is None:
        return False
    if i == 0:
        start = cold_start_time(weekday=state.weekday, profile=state.profile, rng=rng)
    else:
        assert state.current_time is not None and state.prev_lat is not None and state.prev_lon is not None
        travel = estimate_travel_minutes(
            prev_lat=state.prev_lat,
            prev_lon=state.prev_lon,
            lat=poi.lat,
            lon=poi.lon,
            prev_activity=state.prev_activity,
            activity=activity,
            profile=state.profile,
            cfg=cfg,
        )
        start = state.current_time + timedelta(minutes=travel)
    remaining = tuple(state.activity_chain[i + 1 :])
    duration = estimate_duration(
        activity,
        start=start,
        remaining_activities=remaining,
        profile=state.profile,
        cfg=cfg,
    )
    end = start + timedelta(minutes=duration.duration_minutes)
    visit = Visit(
        activity=activity,
        poi_id=poi.poi_id,
        lat=poi.lat,
        lon=poi.lon,
        start=start,
        end=end,
    )
    ok, _ = verify_visit_bounds(visit, cfg=cfg)
    if not ok:
        return False
    state.visits.append(visit)
    state.current_time = end
    state.prev_lat = poi.lat
    state.prev_lon = poi.lon
    state.prev_activity = activity
    state.visit_index += 1
    state.tool_calls_completed += 2
    return True


def run_worker_workflow(
    activity_chain: tuple[str, ...],
    *,
    individual_id: str = "u_001",
    date: str = "2024-06-10",
    weekday: str = "Monday",
    day_type: str = "weekday",
    seed: int = 42,
    cfg: TrajGenAgentConfig | None = None,
    profile: IndividualProfile | None = None,
    peer_profiles: Sequence[IndividualProfile] = (),
) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    rng = random.Random(seed)
    state = WorkflowState(
        individual_id=individual_id,
        date=date,
        weekday=weekday,
        day_type=day_type,
        activity_chain=activity_chain,
        tool_calls_required=len(activity_chain) * 2,
        profile=profile,
        peer_profiles=tuple(peer_profiles),
    )
    while state.visit_index < len(activity_chain):
        if not _ground_visit(state, rng=rng, cfg=cfg):
            break
    traj = DailyTrajectory(
        individual_id=individual_id,
        date=date,
        weekday=weekday,
        visits=tuple(state.visits),
    )
    visit_success = state.tool_calls_completed / max(1, state.tool_calls_required)
    traj_success = 1.0 if len(state.visits) == len(activity_chain) else 0.0
    return {
        "trajectory": traj,
        "n_visits": len(state.visits),
        "trajectory_success": traj_success,
        "visit_success": visit_success,
        "tool_calls_completed": state.tool_calls_completed,
        "tool_calls_required": state.tool_calls_required,
    }


def generate_daily_trajectory(
    *,
    individual_id: str = "u_001",
    date: str = "2024-06-10",
    weekday: str = "Monday",
    day_type: str = "weekday",
    seed: int = 42,
    cfg: TrajGenAgentConfig | None = None,
    use_peer_pool: bool = True,
    reference_corpus: dict[str, tuple[DailyTrajectory, ...]] | None = None,
) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    corpus = reference_corpus or reference_corpus_for_dataset(
        cfg.default_dataset,
        seed=seed,
        n_users=min(cfg.reference_users, 12),
        cfg=cfg,
    )
    profile = profile_for_individual(corpus, individual_id)
    peer_profiles: Sequence[IndividualProfile] = ()
    if use_peer_pool:
        ctx = build_target_context(corpus, target_id=individual_id if individual_id in corpus else "u_001", cfg=cfg)
        profile = ctx["profile"]
        peer_profiles = ctx["peer_profiles"]
    evidence_days = select_evidence_days(corpus.get(individual_id, corpus["u_001"]), weekday=weekday, day_type=day_type)
    historical_chains = [tuple(v.activity for v in t.visits) for t in evidence_days]
    stage1 = generate_activity_chain(
        weekday=weekday,
        day_type=day_type,
        historical_chains=historical_chains,
        profile=profile,
        cfg=cfg,
    )
    chain = tuple(stage1["chain"])
    stage2: dict[str, object] = {"trajectory_success": 0.0}
    for attempt in range(cfg.orchestrator_max_retries + 1):
        stage2 = run_worker_workflow(
            chain,
            individual_id=individual_id,
            date=date,
            weekday=weekday,
            day_type=day_type,
            seed=seed + attempt * 17,
            cfg=cfg,
            profile=profile,
            peer_profiles=peer_profiles,
        )
        if stage2["trajectory_success"] == 1.0:
            break
    return {"stage1": stage1, "stage2": stage2, "activity_chain": chain, "profile_id": profile.individual_id}


def select_evidence_days(
    trajectories: Sequence[DailyTrajectory],
    *,
    weekday: str,
    day_type: str = "weekday",
) -> tuple[DailyTrajectory, ...]:
    from ltx_trainer.trajgenagent.evidence import select_evidence_days as _select

    return _select(trajectories, weekday=weekday, day_type=day_type)
