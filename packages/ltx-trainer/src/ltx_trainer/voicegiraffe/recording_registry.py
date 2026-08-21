"""Full 123-recording catalog and synthetic 1,500-item QA pool (VOICEGIRAFFE §3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.voicegiraffe.config import Domain, MultiHopTask, SingleHopTask, TaskTier, VoiceGiraffeConfig
from ltx_trainer.voicegiraffe.dataset import QAItem, load_qa_items

_DOMAINS = [d.value for d in Domain]
_SINGLE_TASKS = [t.value for t in SingleHopTask]
_MULTI_TASKS = [t.value for t in MultiHopTask]


def build_recording_catalog(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> list[dict[str, Any]]:
    """Deterministic 123-recording manifest matching paper scale stats."""
    cfg = cfg or VoiceGiraffeConfig()
    rng = np.random.default_rng(seed)
    n = cfg.n_recordings
    n_over_hour = int(round(n * cfg.pct_over_one_hour))
    over_flags = np.array([True] * n_over_hour + [False] * (n - n_over_hour))
    rng.shuffle(over_flags)

    catalog: list[dict[str, Any]] = []
    per_domain = max(1, n // len(_DOMAINS))
    for i in range(n):
        domain = _DOMAINS[i % len(_DOMAINS)]
        language = cfg.languages[i % len(cfg.languages)]
        if over_flags[i]:
            duration_min = float(rng.uniform(60.5, 72.0))
        else:
            duration_min = float(rng.uniform(48.0, 59.5))
        catalog.append(
            {
                "recording_id": f"rec-{domain}-{i + 1:03d}",
                "domain": domain,
                "language": language,
                "duration_min": round(duration_min, 1),
                "n_qa_items": 0,
                "domain_bucket": i // per_domain,
            }
        )
    return catalog


def _synthetic_qa_item(
    recording: dict[str, Any],
    *,
    idx: int,
    tier: str,
    task: str,
    rng: np.random.Generator,
) -> QAItem:
    letters = ["A", "B", "C", "D"]
    gold = str(rng.choice(letters))
    choices = {L: f"{task.replace('_', ' ')} option {L} for {recording['domain']}" for L in letters}
    ts = None
    if task == SingleHopTask.TEMPORAL_LOCALIZATION.value:
        ts = float(rng.uniform(60.0, recording["duration_min"] * 60 - 60.0))
        choices = {
            L: f"[{int(ts + off)}, {int(ts + off + 30)}]"
            for L, off in zip(letters, [-120, -30, 0, 90], strict=True)
        }
        gold = "C"
    return QAItem(
        id=f"vg-syn-{idx:04d}",
        recording_id=recording["recording_id"],
        tier=tier,
        task=task,
        domain=recording["domain"],
        language=recording["language"],
        question=f"({tier}/{task}) What is correct for {recording['recording_id']}?",
        choices=choices,
        gold=gold,
        duration_min=float(recording["duration_min"]),
        timestamp_s=ts,
    )


def build_synthetic_qa_pool(
    catalog: list[dict[str, Any]],
    cfg: VoiceGiraffeConfig | None = None,
    *,
    seed: int = 42,
) -> list[QAItem]:
    """Generate synthetic QA to reach paper totals (single + multi hop)."""
    cfg = cfg or VoiceGiraffeConfig()
    rng = np.random.default_rng(seed + 1)
    builtin = load_qa_items()
    need = cfg.n_qa_total - len(builtin)
    n_single = cfg.n_single_hop - sum(1 for i in builtin if i.tier == TaskTier.SINGLE_HOP.value)
    n_multi = cfg.n_multi_hop - sum(1 for i in builtin if i.tier == TaskTier.MULTI_HOP.value)
    n_single = max(0, n_single)
    n_multi = max(0, n_multi)
    # If builtin already covers counts, still pad to n_qa_total
    if n_single + n_multi < need:
        n_single += need - (n_single + n_multi)

    items: list[QAItem] = []
    idx = 0
    for _ in range(n_single):
        rec = catalog[idx % len(catalog)]
        task = _SINGLE_TASKS[idx % len(_SINGLE_TASKS)]
        items.append(_synthetic_qa_item(rec, idx=idx, tier=TaskTier.SINGLE_HOP.value, task=task, rng=rng))
        idx += 1
    for _ in range(n_multi):
        rec = catalog[idx % len(catalog)]
        task = _MULTI_TASKS[idx % len(_MULTI_TASKS)]
        items.append(_synthetic_qa_item(rec, idx=idx, tier=TaskTier.MULTI_HOP.value, task=task, rng=rng))
        idx += 1
    return builtin + items[: max(0, need)]


def attach_qa_counts(catalog: list[dict[str, Any]], qa_items: list[QAItem]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in qa_items:
        counts[item.recording_id] = counts.get(item.recording_id, 0) + 1
    out: list[dict[str, Any]] = []
    for rec in catalog:
        row = dict(rec)
        row["n_qa_items"] = counts.get(rec["recording_id"], 0)
        out.append(row)
    return out


def load_full_qa_pool(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> list[QAItem]:
    cfg = cfg or VoiceGiraffeConfig()
    catalog = build_recording_catalog(cfg, seed=seed)
    return build_synthetic_qa_pool(catalog, cfg, seed=seed)


def catalog_summary(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    durations = np.array([r["duration_min"] for r in catalog], dtype=np.float64)
    return {
        "n_recordings": len(catalog),
        "mean_duration_min": round(float(durations.mean()), 2),
        "pct_over_60_min": round(float(np.mean(durations > 60.0)), 3),
        "domains": sorted({r["domain"] for r in catalog}),
        "languages": sorted({r["language"] for r in catalog}),
    }


def recording_registry_smoke(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    catalog = build_recording_catalog(cfg, seed=seed)
    qa = load_full_qa_pool(cfg, seed=seed)
    catalog = attach_qa_counts(catalog, qa)
    summary = catalog_summary(catalog)
    single = sum(1 for q in qa if q.tier == TaskTier.SINGLE_HOP.value)
    multi = sum(1 for q in qa if q.tier == TaskTier.MULTI_HOP.value)
    return {
        "n_recordings": summary["n_recordings"],
        "n_qa_total": len(qa),
        "n_single_hop": single,
        "n_multi_hop": multi,
        "mean_duration_min": summary["mean_duration_min"],
        "pct_over_60_min": summary["pct_over_60_min"],
        "matches_paper_scale": summary["n_recordings"] == cfg.n_recordings and len(qa) == cfg.n_qa_total,
    }
