"""Compare experiment outputs to LiveSVG paper tables (arXiv:2605.30174)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.livesvg.paper_tables import (
    human_preference_rates,
    table_aniclipart_quantitative,
    table_challengesvg_quantitative,
    table_runtime_minutes,
)


def compare_fitting_run(experiment: dict[str, Any]) -> dict[str, Any]:
    """Smoke checks for synthetic / local fitting runs."""
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "check": "loss_decreased",
            "ok": bool(experiment.get("loss_decreased")),
            "got": experiment.get("loss_last"),
            "first": experiment.get("loss_first"),
        }
    )
    mse = float(experiment.get("per_frame_mse_mean", 1.0))
    is_synthetic = experiment.get("source") == "synthetic"
    mse_limit = 0.99 if is_synthetic else 0.5
    checks.append(
        {
            "check": "per_frame_mse_below_threshold",
            "ok": mse < mse_limit,
            "got": mse,
            "limit": mse_limit,
        }
    )
    active = int(experiment.get("active_keyframes_end", 1))
    n_kf = int(experiment.get("num_keyframes", 15))
    checks.append(
        {
            "check": "progressive_activation",
            "ok": active >= min(3, n_kf),
            "got": active,
            "expected_min": min(3, n_kf),
        }
    )
    passed = sum(1 for c in checks if c.get("ok"))
    return {"checks": checks, "passed": passed, "total": len(checks), "all_ok": passed == len(checks)}


def compare_metrics_to_paper(
    metrics: dict[str, float],
    *,
    benchmark: str = "aniclipart",
    method: str = "LiveSVG (Veo 3.1)",
) -> dict[str, Any]:
    """Compare reported XCLIP / LPIPS / SSIM to published Table 2 or 4."""
    tables = {
        "aniclipart": table_aniclipart_quantitative(),
        "challengesvg": table_challengesvg_quantitative(),
    }
    ref = tables.get(benchmark, {}).get(method, {})
    checks: list[dict[str, Any]] = []
    for key in ("XCLIP", "LPIPS", "SSIM", "DOVER"):
        if key not in metrics or key not in ref:
            continue
        got, expected = float(metrics[key]), float(ref[key])
        tol = 0.05 if key == "XCLIP" else 0.15
        checks.append({"metric": key, "ok": abs(got - expected) <= tol, "got": got, "ref": expected})
    passed = sum(1 for c in checks if c.get("ok"))
    return {"method": method, "benchmark": benchmark, "checks": checks, "passed": passed, "total": len(checks)}


def compare_experiment_to_paper(experiment: dict[str, Any]) -> dict[str, Any]:
    """Full report: fitting behavior + optional metric dict in experiment."""
    fitting = compare_fitting_run(experiment)
    sections: dict[str, Any] = {"fitting": fitting}
    if "metrics" in experiment:
        sections["metrics"] = compare_metrics_to_paper(experiment["metrics"])
    prefs = human_preference_rates()
    sections["paper_claims"] = {
        "aniclipart_aggregate_win_pct": prefs["AniClipart_overall"]["LiveSVG_aggregate"],
        "challengesvg_win_pct": prefs["ChallengeSVG_overall_win_pct"],
        "runtime_minutes_veo": table_runtime_minutes()["LiveSVG (Veo 3.1)"],
    }
    all_ok = fitting.get("all_ok", False)
    if "metrics" in sections:
        all_ok = all_ok and sections["metrics"].get("passed", 0) == sections["metrics"].get("total", 0)
    return {
        "sections": sections,
        "all_ok": all_ok,
        "paper": "arXiv:2605.30174",
    }


def format_report_markdown(report: dict[str, Any]) -> str:
    lines = [f"# LiveSVG paper report ({report.get('paper', '')})", ""]
    lines.append(f"**Overall:** {'PASS' if report.get('all_ok') else 'PARTIAL'}")
    lines.append("")
    for name, sec in (report.get("sections") or {}).items():
        lines.append(f"## {name}")
        if "checks" in sec:
            for c in sec["checks"]:
                mark = "ok" if c.get("ok") else "fail"
                lines.append(f"- [{mark}] {c.get('check') or c.get('metric')}: {c}")
        else:
            lines.append(f"```json\n{sec}\n```")
        lines.append("")
    return "\n".join(lines)
