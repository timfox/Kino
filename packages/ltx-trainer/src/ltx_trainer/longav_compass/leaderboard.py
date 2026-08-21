"""Leaderboard helpers from paper table excerpts."""

from __future__ import annotations

from typing import Any

from ltx_trainer.longav_compass.metrics import balanced_score
from ltx_trainer.longav_compass.tables import (
    table_i2av_main_results,
    table_t2av_main_results,
    table_v2av_main_results,
)


def _row_balanced_t2av(row: dict[str, Any]) -> float:
    metrics = {
        "VQA": float(row["VQA"]),
        "VQ": float(row["VQ"]),
        "Cont.": float(row["Cont."]),
        "Hol.": float(row["Hol."]),
        "TVAlign": float(row["TVAlign"]),
    }
    if row.get("AVS") is not None:
        metrics["AVS"] = float(row["AVS"])
        metrics["AudQ"] = float(row["AudQ"])
        metrics["AudL"] = float(row["AudL"])
    return balanced_score(metrics)


def rank_models_t2av(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Rank T2AV models by simplified balanced score (Table 3 excerpt)."""
    rows = rows or table_t2av_main_results()
    ranked = sorted(
        (
            {
                "model": r["model"],
                "balanced_score": _row_balanced_t2av(r),
                "VQA": r["VQA"],
                "Cont.": r["Cont."],
                "audio": r.get("audio", False),
            }
            for r in rows
        ),
        key=lambda x: x["balanced_score"],
        reverse=True,
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def rank_models_i2av(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = rows or table_i2av_main_results()
    ranked = sorted(
        rows,
        key=lambda r: balanced_score(
            {
                "VQA": r["VQA"],
                "VQ": r["VQ"],
                "Cont.": r["Cont."],
                "ImgAlign": r["ImgAlign"],
                "AVS": r.get("AVS", 3.0),
            }
        ),
        reverse=True,
    )
    return [{"rank": i + 1, **r} for i, r in enumerate(ranked)]


def rank_models_v2av(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = rows or table_v2av_main_results()
    ranked = sorted(
        rows,
        key=lambda r: balanced_score(
            {"VQA": r["VQA"], "VQ": r["VQ"], "Cont.": r["Cont."], "TVAlign": r.get("TVAlign", 0.6)}
        ),
        reverse=True,
    )
    return [{"rank": i + 1, **r} for i, r in enumerate(ranked)]


def diagnostic_gaps_for_model(model: str, task: str = "T2AV") -> dict[str, Any]:
    """Highlight where a model trails the task leader (paper Sec. 4.3–4.4)."""
    task = task.upper()
    if task == "T2AV":
        rows = table_t2av_main_results()
        leader = max(rows, key=_row_balanced_t2av)
    elif task == "I2AV":
        rows = table_i2av_main_results()
        leader = max(
            rows,
            key=lambda r: balanced_score(
                {"VQA": r["VQA"], "VQ": r["VQ"], "Cont.": r["Cont."], "ImgAlign": r["ImgAlign"]}
            ),
        )
    else:
        rows = table_v2av_main_results()
        leader = max(
            rows,
            key=lambda r: balanced_score(
                {"VQA": r["VQA"], "VQ": r["VQ"], "Cont.": r["Cont."], "TVAlign": r.get("TVAlign", 0.6)}
            ),
        )
    target = next(r for r in rows if r["model"] == model)
    gaps: dict[str, float] = {}
    for key in set(leader) & set(target):
        if key in ("model", "audio"):
            continue
        lv, tv = leader.get(key), target.get(key)
        if isinstance(lv, (int, float)) and isinstance(tv, (int, float)) and lv > tv:
            gaps[key] = round(float(lv) - float(tv), 4)
    return {
        "task": task,
        "model": model,
        "leader": leader["model"],
        "gaps_vs_leader": gaps,
        "notes": _diagnostic_notes(model, gaps),
    }


def _diagnostic_notes(model: str, gaps: dict[str, float]) -> list[str]:
    notes: list[str] = []
    if gaps.get("VQA", 0) > 0.05:
        notes.append("Event fulfillment below leader — script following / missing events.")
    if gaps.get("Cont.", 0) > 0.3:
        notes.append("Long-form continuity weak — cross-event identity or narrative drift.")
    if gaps.get("ImgAlign", 0) > 0.05:
        notes.append("Reference image similarity high but fulfillment may still fail (paper I2AV finding).")
    if model == "VideoDirectorGPT":
        notes.append("High ImgAlign with low VQA/Cont. — appearance preservation ≠ minute-long I2AV success.")
    if not notes:
        notes.append("No large gaps vs task leader on excerpted metrics.")
    return notes


def leaderboard_bundle() -> dict[str, Any]:
    return {
        "T2AV": rank_models_t2av(),
        "I2AV": rank_models_i2av(),
        "V2AV": rank_models_v2av(),
        "seedance_t2av_gaps": diagnostic_gaps_for_model("LTX 2.3", "T2AV"),
        "videodirector_i2av_gaps": diagnostic_gaps_for_model("VideoDirectorGPT", "I2AV"),
    }
