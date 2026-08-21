"""NumoSim / MobilitySyn stay-point ingest (Sec. III-B)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from ltx_trainer.trajgenagent.profiles import build_profile_from_trajectories, profile_summary
from ltx_trainer.trajgenagent.trajectory import DailyTrajectory, Visit


@dataclass(frozen=True)
class StayPointRecord:
    individual_id: str
    date: str
    weekday: str
    activity: str
    poi_id: str
    lat: float
    lon: float
    start: datetime
    end: datetime


def parse_staypoint_row(row: dict[str, Any]) -> StayPointRecord:
    """Parse one stay-point dict (activity, POI, ts, te)."""
    start = datetime.fromisoformat(str(row["ts"]))
    end = datetime.fromisoformat(str(row["te"]))
    return StayPointRecord(
        individual_id=str(row["individual_id"]),
        date=str(row.get("date", start.date().isoformat())),
        weekday=str(row.get("weekday", start.strftime("%A"))),
        activity=str(row["activity"]),
        poi_id=str(row["poi_id"]),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        start=start,
        end=end,
    )


def group_daily_trajectories(records: Sequence[StayPointRecord]) -> list[DailyTrajectory]:
    buckets: dict[tuple[str, str], list[Visit]] = {}
    meta: dict[tuple[str, str], tuple[str, str]] = {}
    for rec in records:
        key = (rec.individual_id, rec.date)
        meta[key] = (rec.individual_id, rec.weekday)
        buckets.setdefault(key, []).append(
            Visit(
                activity=rec.activity,
                poi_id=rec.poi_id,
                lat=rec.lat,
                lon=rec.lon,
                start=rec.start,
                end=rec.end,
            )
        )
    out: list[DailyTrajectory] = []
    for key, visits in buckets.items():
        individual_id, weekday = meta[key]
        visits.sort(key=lambda v: v.start)
        out.append(
            DailyTrajectory(
                individual_id=individual_id,
                date=key[1],
                weekday=weekday,
                visits=tuple(visits),
            )
        )
    return out


def load_staypoints_jsonl(path: str | Path) -> tuple[StayPointRecord, ...]:
    """Load stay-point rows from JSONL (one dict per line)."""
    records: list[StayPointRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(parse_staypoint_row(json.loads(line)))
    return tuple(records)


def ingest_from_jsonl(path: str | Path) -> dict[str, object]:
    """Build trajectories + individual profiles from a stay-point JSONL file."""
    records = load_staypoints_jsonl(path)
    trajectories = group_daily_trajectories(records)
    by_user: dict[str, list[DailyTrajectory]] = {}
    for traj in trajectories:
        by_user.setdefault(traj.individual_id, []).append(traj)
    profiles = {
        uid: build_profile_from_trajectories(uid, trajs)
        for uid, trajs in by_user.items()
    }
    return {
        "path": str(path),
        "n_records": len(records),
        "n_trajectories": len(trajectories),
        "n_individuals": len(profiles),
        "profiles": {uid: profile_summary(profiles[uid]) for uid in profiles},
        "trajectories": trajectories,
    }


def ingest_demo_rows() -> tuple[StayPointRecord, ...]:
    """Minimal LA-style weekday chain for ingest smoke."""
    base = datetime(2024, 6, 10, 8, 0, 0)
    rows: list[StayPointRecord] = []
    chain = (
        ("Home", "poi_h1", 34.05, -118.25, 480),
        ("Work", "poi_w1", 34.06, -118.24, 240),
        ("EatOut", "poi_e1", 34.055, -118.245, 45),
        ("Work", "poi_w1", 34.06, -118.24, 180),
        ("Home", "poi_h1", 34.05, -118.25, 600),
    )
    t = base
    for activity, poi_id, lat, lon, minutes in chain:
        end = t + timedelta(minutes=minutes)
        rows.append(
            StayPointRecord(
                individual_id="u_001",
                date="2024-06-10",
                weekday="Monday",
                activity=activity,
                poi_id=poi_id,
                lat=lat,
                lon=lon,
                start=t,
                end=end,
            )
        )
        t = end
    return tuple(rows)


def ingest_demo() -> dict[str, object]:
    records = ingest_demo_rows()
    trajectories = group_daily_trajectories(records)
    traj = trajectories[0]
    profile = build_profile_from_trajectories("u_001", trajectories)
    return {
        "n_records": len(records),
        "n_trajectories": len(trajectories),
        "n_visits": traj.n_visits,
        "starts_home": traj.visits[0].activity == "Home",
        "ends_home": traj.visits[-1].activity == "Home",
        "profile_n_transitions": len(profile.transition_freq),
    }
