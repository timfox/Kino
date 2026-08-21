"""Paper subset filters and Table 2(b) complexity / modality reporting."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

from ltx_trainer.mmae.benchmarks import TABLE2_BY_COMPLEXITY, TABLE2_MIXED_MODALITY, TABLE2_OVERALL
from ltx_trainer.mmae.constants import MMAE_SHORT_SUBSET_COUNT, MMAE_SHORT_SUBSET_MAX_SEC
from ltx_trainer.mmae.editors import table2_delta
from ltx_trainer.mmae.metrics import aggregate_rates_pct
from ltx_trainer.mmae.sample import MMAESample

MULTIPLE_COMPLEXITIES = frozenset(
    {
        "multiple",
        "multi-instruction",
        "multi-part",
        "multi-hop",
        "multi-round",
        "multi-audio",
    }
)


def complexity_bucket(complexity: str) -> str:
    """Map MMAE complexity label to Table 2(b) ``single`` vs ``multiple`` bucket."""
    if complexity == "single":
        return "single"
    return "multiple"


def table2_complexity_anchor(model: str, bucket: str) -> dict[str, float] | None:
    by_model = TABLE2_BY_COMPLEXITY.get(model)
    if not by_model:
        return None
    return by_model.get(bucket)


def table2_modality_anchor(model: str, modality: str) -> dict[str, float] | None:
    by_model = TABLE2_MIXED_MODALITY.get(model)
    if not by_model:
        return None
    return by_model.get(modality)


def probe_wav_duration_sec(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                return None
            return handle.getnframes() / float(rate)
    except (OSError, wave.Error):
        return None


def build_duration_index(
    samples: list[MMAESample],
    *,
    dataset_root: str | Path,
) -> dict[str, float]:
    """Longest resolved input WAV duration per sample (seconds)."""
    root = Path(dataset_root)
    out: dict[str, float] = {}
    for sample in samples:
        durations: list[float] = []
        for rel in sample.audio_paths:
            path = root / rel
            if not path.is_file():
                continue
            dur = probe_wav_duration_sec(path)
            if dur is not None:
                durations.append(dur)
        if durations:
            out[sample.sample_id] = max(durations)
    return out


def filter_short_subset(
    samples: list[MMAESample],
    *,
    dataset_root: str | Path | None = None,
    max_duration_sec: float = MMAE_SHORT_SUBSET_MAX_SEC,
    durations: dict[str, float] | None = None,
) -> tuple[list[MMAESample], dict[str, float]]:
    """Paper ≤10s subset (801 samples when fully cached on Hub WAV tree)."""
    from ltx_trainer.mmae.evaluate import filter_samples_by_duration

    if durations is None and dataset_root is not None:
        durations = build_duration_index(samples, dataset_root=dataset_root)
    kept = filter_samples_by_duration(samples, max_duration_sec, durations=durations or {})
    return kept, durations or {}


def breakdown_by_complexity_bucket(
    per_sample: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = {}
    for row in per_sample:
        key = complexity_bucket(str(row.get("complexity", "single")))
        buckets.setdefault(key, []).append(
            {k: float(row[k]) for k in ("IFR", "CR", "EMR") if k in row}
        )
    return {k: aggregate_rates_pct(v) for k, v in sorted(buckets.items())}


def table2_complexity_deltas(
    model: str,
    by_bucket: dict[str, dict[str, float]],
) -> dict[str, dict[str, float] | None]:
    out: dict[str, dict[str, float] | None] = {}
    for bucket in ("single", "multiple"):
        rates = by_bucket.get(bucket)
        anchor = table2_complexity_anchor(model, bucket)
        if rates is None or anchor is None:
            out[bucket] = None
            continue
        out[bucket] = {
            key: round(float(rates[key]) - float(anchor[key]), 2)
            for key in ("IFR", "CR", "EMR")
            if key in rates and key in anchor
        }
    return out


def table2_modality_deltas(
    model: str,
    by_modality: dict[str, dict[str, float]],
) -> dict[str, dict[str, float] | None]:
    out: dict[str, dict[str, float] | None] = {}
    for modality, anchor in TABLE2_MIXED_MODALITY.get(model, {}).items():
        rates = by_modality.get(modality)
        if rates is None:
            out[modality] = None
            continue
        out[modality] = {
            key: round(float(rates[key]) - float(anchor[key]), 2)
            for key in ("IFR", "CR", "EMR")
            if key in rates and key in anchor
        }
    return out


def enrich_evaluation_report(
    evaluation: dict[str, Any],
    *,
    model: str,
    short_subset: bool = False,
    short_count: int | None = None,
) -> dict[str, Any]:
    """Attach Table 2(b) buckets, mixed-modality deltas, and optional short-subset note."""
    per_sample = evaluation.get("samples") or []
    by_bucket = breakdown_by_complexity_bucket(per_sample)
    evaluation = dict(evaluation)
    evaluation["by_complexity_table2"] = by_bucket
    evaluation["table2_complexity_delta"] = table2_complexity_deltas(model, by_bucket)
    by_modality = evaluation.get("by_modality") or {}
    evaluation["table2_modality_delta"] = table2_modality_deltas(model, by_modality)
    evaluation["table2_overall_anchor"] = TABLE2_OVERALL.get(model)
    evaluation["table2_overall_delta"] = (
        table2_delta(model, evaluation["rates_pct"]) if model in TABLE2_OVERALL else None
    )
    if short_subset:
        evaluation["short_subset"] = {
            "max_duration_sec": MMAE_SHORT_SUBSET_MAX_SEC,
            "paper_count": MMAE_SHORT_SUBSET_COUNT,
            "eval_count": short_count if short_count is not None else evaluation.get("num_samples"),
        }
    return evaluation
