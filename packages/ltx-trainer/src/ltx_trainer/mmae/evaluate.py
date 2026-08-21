"""Batch MMAE evaluation: mock judger, submission layout, optional OpenAI judger."""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from ltx_trainer.mmae.audio_bundle import clips_for_rubric
from ltx_trainer.mmae.catalog import synthetic_catalog
from ltx_trainer.mmae.config import MMAEConfig
from ltx_trainer.mmae.judger_backend import OmniSampleJudger, resolve_judger_config
from ltx_trainer.mmae.metrics import aggregate_rates_pct, rates_from_rubric_scores
from ltx_trainer.mmae.paper_subset import (
    complexity_bucket,
    enrich_evaluation_report,
    filter_short_subset,
)
from ltx_trainer.mmae.paths import resolve_prediction_wav
from ltx_trainer.mmae.rubrics import score_sample
from ltx_trainer.mmae.sample import MMAESample
from ltx_trainer.mmae.simulation import model_skill_from_table


def resolve_dataset_wav(root: Path, rel_path: str) -> Path | None:
    path = root / rel_path
    return path if path.is_file() else None


def audio_context_note(
    sample: MMAESample,
    *,
    dataset_root: Path | None,
    predictions_dir: Path | None,
) -> str:
    lines: list[str] = []
    if predictions_dir is not None:
        out = resolve_prediction_wav(predictions_dir, sample.sample_id)
        if out:
            lines.append(f"output: {out}")
    if dataset_root is not None:
        for i, rel in enumerate(sample.audio_paths, start=1):
            resolved = resolve_dataset_wav(dataset_root, rel)
            if resolved:
                lines.append(f"input{i}: {resolved}")
    return "\n".join(lines)


def filter_samples_by_duration(
    samples: list[MMAESample],
    max_duration_sec: float,
    *,
    durations: dict[str, float] | None = None,
) -> list[MMAESample]:
    """Keep samples with known duration ≤ *max_duration_sec* (paper ≤10s subset)."""
    durations = durations or {}
    kept: list[MMAESample] = []
    for sample in samples:
        dur = durations.get(sample.sample_id)
        if dur is None and sample.duration_sec is not None:
            dur = sample.duration_sec
        if dur is None:
            continue
        if dur <= max_duration_sec:
            kept.append(sample)
    return kept


def submission_layout_summary(
    predictions_dir: str | Path,
    samples: list[MMAESample] | None = None,
) -> dict[str, Any]:
    root = Path(predictions_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"predictions directory not found: {root}")
    samples = samples or synthetic_catalog()
    present: list[str] = []
    missing: list[str] = []
    for sample in samples:
        if resolve_prediction_wav(root, sample.sample_id):
            present.append(sample.sample_id)
        else:
            missing.append(sample.sample_id)
    return {
        "predictions_dir": str(root),
        "num_samples": len(samples),
        "present": present,
        "missing": missing,
        "coverage": len(present) / max(len(samples), 1),
    }


def evaluate_samples(
    samples: list[MMAESample],
    *,
    model: str = "Step-Audio-EditX",
    cfg: MMAEConfig | None = None,
    seed: int = 0,
    judger_mode: str | None = None,
    dataset_root: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    short_subset: bool = False,
) -> dict[str, Any]:
    """Score *samples* with mock Table-2 rates or ``GOPEX_MMAE_JUDGER_MODE=openai|omni``."""
    cfg = cfg or MMAEConfig()
    mode = judger_mode or resolve_judger_config().mode
    ds_root = Path(dataset_root) if dataset_root else None
    pred_root = Path(predictions_dir) if predictions_dir else None
    p_if, p_cr = model_skill_from_table(model)

    eval_samples = samples
    duration_index: dict[str, float] | None = None
    if short_subset and ds_root is not None:
        eval_samples, duration_index = filter_short_subset(samples, dataset_root=ds_root)

    cache = None
    if cache_dir is not None and mode in ("openai", "omni"):
        from ltx_trainer.mmae.judger_cache import JudgerVoteCache

        cache = JudgerVoteCache(cache_dir)

    per_sample: list[dict[str, Any]] = []
    for i, sample in enumerate(eval_samples):
        if mode == "openai":
            from ltx_trainer.mmae.judger_backend import OpenAICompatTextJudger

            cfg_j = resolve_judger_config()
            note = audio_context_note(sample, dataset_root=ds_root, predictions_dir=pred_root)
            active_judger = OpenAICompatTextJudger(
                base_url=cfg_j.base_url or "",
                model=cfg_j.model or "",
                api_key=cfg_j.api_key,
                timeout_sec=cfg_j.timeout_sec,
                audio_note=note,
            )
            if cache is not None:
                from ltx_trainer.mmae.judger_cache import score_sample_cached

                scored = score_sample_cached(
                    sample,
                    judger=active_judger,
                    cache=cache,
                    cfg=cfg,
                    seed=seed + i,
                    resume=resume,
                )
            else:
                scored = score_sample(sample, judger=active_judger, cfg=cfg, seed=seed + i)
        elif mode == "omni":
            cfg_j = resolve_judger_config()
            active_judger = OmniSampleJudger(
                base_url=cfg_j.base_url or "",
                model=cfg_j.model or "",
                sample=sample,
                dataset_root=ds_root,
                predictions_dir=pred_root,
                api_key=cfg_j.api_key,
                timeout_sec=cfg_j.timeout_sec,
                part_style=cfg_j.omni_part_style,
                require_audio=cfg_j.omni_require_audio,
                slice_cache_dir=Path(tempfile.gettempdir()) / "gopex_mmae_slices" / sample.sample_id,
            )
            if cache is not None:
                from ltx_trainer.mmae.judger_cache import score_sample_cached

                scored = score_sample_cached(
                    sample,
                    judger=active_judger,
                    cache=cache,
                    cfg=cfg,
                    seed=seed + i,
                    resume=resume,
                )
            else:
                scored = score_sample(sample, judger=active_judger, cfg=cfg, seed=seed + i)
        else:
            scored = score_sample(
                sample,
                p_correct_if=p_if,
                p_correct_cr=p_cr,
                cfg=cfg,
                seed=seed + i,
            )
        rates = rates_from_rubric_scores(sample, scored["rubrics"])
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "complexity": sample.complexity,
                "complexity_bucket": complexity_bucket(sample.complexity),
                "modality": sample.modality,
                **rates,
            }
        )
    agg = aggregate_rates_pct([{k: v for k, v in s.items() if k in ("IFR", "CR", "EMR")} for s in per_sample])
    result = {
        "model": model,
        "judger_mode": mode,
        "num_samples": len(eval_samples),
        "rates_pct": agg,
        "by_complexity": breakdown_by_field(per_sample, "complexity"),
        "by_complexity_bucket": breakdown_by_field(per_sample, "complexity_bucket"),
        "by_modality": breakdown_by_field(per_sample, "modality"),
        "samples": per_sample,
    }
    if cache is not None:
        result["vote_cache"] = {"dir": str(cache.cache_dir), **cache.stats()}
    if short_subset:
        result["short_subset"] = {
            "enabled": True,
            "duration_index_size": len(duration_index or {}),
        }
    return enrich_evaluation_report(
        result,
        model=model,
        short_subset=short_subset,
        short_count=len(eval_samples),
    )


def breakdown_by_field(
    per_sample: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in per_sample:
        key = str(row.get(field, "unknown"))
        buckets[key].append({k: float(row[k]) for k in ("IFR", "CR", "EMR") if k in row})
    return {k: aggregate_rates_pct(v) for k, v in sorted(buckets.items())}


def evaluate_catalog(
    *,
    model: str = "Step-Audio-EditX",
    cfg: MMAEConfig | None = None,
    seed: int = 0,
    judger_mode: str | None = None,
) -> dict[str, Any]:
    return evaluate_samples(
        synthetic_catalog(),
        model=model,
        cfg=cfg,
        seed=seed,
        judger_mode=judger_mode,
    )


def evaluate_submission(
    predictions_dir: str | Path,
    samples: list[MMAESample] | None = None,
    *,
    model: str = "Step-Audio-EditX",
    cfg: MMAEConfig | None = None,
    seed: int = 0,
    require_all: bool = False,
    judger_mode: str | None = None,
    dataset_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    short_subset: bool = False,
) -> dict[str, Any]:
    """Check submission layout then score rubrics (mock or OpenAI text judger)."""
    samples = samples or synthetic_catalog()
    layout = submission_layout_summary(predictions_dir, samples)
    if require_all and layout["missing"]:
        raise FileNotFoundError(f"missing predictions for: {layout['missing']}")
    scored = evaluate_samples(
        samples,
        model=model,
        cfg=cfg,
        seed=seed,
        judger_mode=judger_mode,
        dataset_root=dataset_root,
        predictions_dir=predictions_dir,
        cache_dir=cache_dir,
        resume=resume,
        short_subset=short_subset,
    )
    mode = scored["judger_mode"]
    from ltx_trainer.mmae.benchmarks import TABLE2_OVERALL
    from ltx_trainer.mmae.editors import table2_delta

    anchor = TABLE2_OVERALL.get(model)
    delta = table2_delta(model, scored["rates_pct"]) if anchor else None
    if mode == "omni":
        note = "Qwen3-Omni multimodal judger with attached WAV clips (Appendix C)."
    elif mode == "openai":
        note = "OpenAI-compat text judger (audio paths in prompt only)."
    else:
        note = "Mock judger uses Table 2 anchor rates."
    return {
        "layout": layout,
        "evaluation": scored,
        "table2_anchor": anchor,
        "table2_delta": delta,
        "table2_complexity_delta": scored.get("table2_complexity_delta"),
        "table2_modality_delta": scored.get("table2_modality_delta"),
        "note": note,
    }


def export_rubric_manifest(
    samples: list[MMAESample],
    output_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write JSONL rubric rows for external batch judgers."""
    ds_root = Path(dataset_root) if dataset_root else None
    pred_root = Path(predictions_dir) if predictions_dir else None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for idx, rubric in enumerate(sample.rubrics):
            clips = clips_for_rubric(
                rubric,
                sample,
                dataset_root=ds_root,
                predictions_dir=pred_root,
            )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "rubric_index": idx,
                    "category": rubric.category.value,
                    "complexity": sample.complexity,
                    "modality": sample.modality,
                    "question": rubric.question,
                    "choices": rubric.all_choices(),
                    "right_choice": rubric.right_choice,
                    "audio_clips": [
                        {
                            "label": clip.label,
                            "path": str(clip.path),
                            "start_sec": clip.start_sec,
                            "end_sec": clip.end_sec,
                        }
                        for clip in clips
                    ],
                }
            )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"output": str(path), "num_rows": len(rows), "num_samples": len(samples)}


def load_samples_for_eval(
    *,
    root: str | Path | None = None,
    limit: int | None = None,
) -> list[MMAESample]:
    from gopex_datasets.mmae.load import load_samples_from_root

    if root is not None:
        return load_samples_from_root(root, limit=limit)
    try:
        return load_samples_from_root(limit=limit)
    except FileNotFoundError:
        return synthetic_catalog()[: limit or len(synthetic_catalog())]


def load_hub_samples(
    *,
    split: str = "full",
    streaming: bool = True,
    limit: int | None = None,
    root: str | Path | None = None,
) -> list[MMAESample]:
    """Load MMAE samples from cached/GitHub ``MMAE-meta.json``."""
    from gopex_datasets.mmae.hub import iter_hub_samples

    return iter_hub_samples(split=split, streaming=streaming, limit=limit, root=root)


def import_rubric_votes_to_cache(
    votes_path: str | Path,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Seed a judger vote cache from external JSONL (``sample_id,rubric_index,vote_index,choice``)."""
    from ltx_trainer.mmae.judger_cache import JudgerVoteCache, import_rubric_votes

    cache = JudgerVoteCache(cache_dir)
    imported = import_rubric_votes(votes_path)
    for key, choice in imported._votes.items():
        cache.set_vote(key[0], key[1], key[2], choice, flush=False)
    if cache._votes:
        with cache.votes_path.open("w", encoding="utf-8") as handle:
            for (sample_id, rubric_index, vote_index), choice in sorted(cache._votes.items()):
                handle.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "rubric_index": rubric_index,
                            "vote_index": vote_index,
                            "choice": choice,
                        }
                    )
                    + "\n"
                )
    return {"cache_dir": str(cache.cache_dir), **cache.stats(), "source": str(votes_path)}


def evaluate_hub(
    *,
    model: str = "Step-Audio-EditX",
    cfg: MMAEConfig | None = None,
    seed: int = 0,
    judger_mode: str | None = None,
    split: str = "full",
    streaming: bool = True,
    limit: int | None = None,
    dataset_root: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    short_subset: bool = False,
) -> dict[str, Any]:
    """Score Hub split samples (streaming by default; use *limit* for smoke)."""
    samples = load_hub_samples(split=split, streaming=streaming, limit=limit, root=dataset_root)
    scored = evaluate_samples(
        samples,
        model=model,
        cfg=cfg,
        seed=seed,
        judger_mode=judger_mode,
        dataset_root=dataset_root,
        predictions_dir=predictions_dir,
        cache_dir=cache_dir,
        resume=resume,
        short_subset=short_subset,
    )
    scored["hub"] = {
        "dataset": "BoJack/MMAE",
        "meta_source": "https://raw.githubusercontent.com/ddlBoJack/MMAE/main/MMAE-meta.json",
        "split": split,
        "streaming": streaming,
        "limit": limit,
    }
    return scored


def write_eval_report(
    report_path: str | Path,
    payload: dict[str, Any],
    *,
    title: str = "MMAE evaluation report",
) -> dict[str, Any]:
    """Persist evaluation JSON and a short Markdown summary beside it."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path if path.suffix == ".json" else path.with_suffix(".json")
    md_path = json_path.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [f"# {title}", ""]
    model = payload.get("model") or payload.get("evaluation", {}).get("model")
    if model:
        lines.append(f"**Model:** {model}")
    judger = payload.get("judger_mode") or payload.get("evaluation", {}).get("judger_mode")
    if judger:
        lines.append(f"**Judger:** {judger}")
    rates = payload.get("rates_pct") or payload.get("evaluation", {}).get("rates_pct")
    if isinstance(rates, dict):
        lines.extend(["", "## Rates (%)", ""])
        for key in ("IFR", "CR", "EMR"):
            if key in rates:
                lines.append(f"- **{key}:** {rates[key]:.2f}")
    anchor = payload.get("table2_anchor")
    delta = payload.get("table2_delta")
    complexity_delta = payload.get("table2_complexity_delta") or payload.get("evaluation", {}).get(
        "table2_complexity_delta"
    )
    if anchor or delta or complexity_delta:
        lines.extend(["", "## Table 2 anchor", ""])
        if anchor:
            lines.append(f"- overall anchor: {anchor}")
        if delta:
            lines.append(f"- overall delta: {delta}")
        if complexity_delta:
            lines.append(f"- complexity delta: {complexity_delta}")
    layout = payload.get("layout")
    if isinstance(layout, dict) and "coverage" in layout:
        lines.extend(["", "## Submission layout", ""])
        lines.append(f"- coverage: {layout['coverage']:.2%}")
        lines.append(f"- present: {len(layout.get('present', []))}")
        lines.append(f"- missing: {len(layout.get('missing', []))}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
