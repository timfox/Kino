"""Batch evaluation helpers for upstream checkouts and model output trees."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ltx_trainer.longav_compass.annotation import BenchmarkCase, case_to_dict
from ltx_trainer.longav_compass.io import (
    discover_case_ids,
    load_case_from_upstream,
    model_output_paths,
    summarize_checkout,
)
from ltx_trainer.longav_compass.judges import evaluate_case


def _existing(paths: dict[str, Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in paths.items():
        if v.is_file():
            out[k] = str(v)
        elif v.is_dir() and any(v.iterdir()):
            out[k] = str(v)
    return out


def evaluate_upstream_case(
    checkout_root: str | Path,
    case_id: str,
    *,
    model_slug: str | None = None,
    has_audio: bool = True,
    quality_bias: float = 0.88,
) -> dict[str, Any]:
    """Load annotation from checkout and run stub hierarchical scoring."""
    case = load_case_from_upstream(checkout_root, case_id)
    report = evaluate_case(case, has_audio=has_audio, quality_bias=quality_bias)
    report["checkout"] = summarize_checkout(checkout_root)
    if model_slug:
        report["artifacts"] = _existing(model_output_paths(checkout_root, case_id, model_slug))
        report["model_slug"] = model_slug
    return report


def evaluate_checkout_batch(
    checkout_root: str | Path,
    *,
    case_ids: list[str] | None = None,
    limit: int = 8,
    has_audio: bool = True,
    quality_bias: float = 0.88,
) -> dict[str, Any]:
    """Evaluate up to ``limit`` cases discovered under ``cases/``."""
    root = Path(checkout_root)
    ids = case_ids or discover_case_ids(root)
    if limit > 0:
        ids = ids[:limit]
    reports = [
        evaluate_case(load_case_from_upstream(root, cid), has_audio=has_audio, quality_bias=quality_bias)
        for cid in ids
    ]
    return {
        "checkout": summarize_checkout(root),
        "evaluated": len(reports),
        "reports": reports,
    }


def evaluate_case_object(
    case: BenchmarkCase,
    *,
    has_audio: bool = True,
    quality_bias: float = 0.88,
) -> dict[str, Any]:
    """Thin wrapper returning case dict alongside scores."""
    return {
        "case": case_to_dict(case),
        "evaluation": evaluate_case(case, has_audio=has_audio, quality_bias=quality_bias),
    }
