"""Compare experiment outputs to published tables (arXiv:2605.24322)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.physics_steering.paper_tables import (
    table_alpha_sweep,
    table_block_cav_disentanglement,
    table_layer_ablation,
    table_probe_accuracy_by_layer,
)


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def compare_alpha_sweep(
    rows: list[dict[str, float]],
    *,
    tol_p: float = 0.15,
    tol_flip: float = 0.35,
) -> dict[str, Any]:
    """Check α-sweep qualitative behavior vs Tab. 2."""
    ref = {int(r["alpha"]): r for r in table_alpha_sweep()}
    got = {int(r["alpha"]): r for r in rows}
    checks: list[dict[str, Any]] = []
    for alpha in (-5, 0, 5):
        if alpha not in ref or alpha not in got:
            continue
        r, g = ref[alpha], got[alpha]
        if alpha == 0:
            ok = g.get("flip_rate", 0) == 0.0
            checks.append({"alpha": alpha, "check": "flip_rate_zero", "ok": ok, "got": g.get("flip_rate")})
            continue
        if alpha > 0:
            ok = g.get("p_impossible", 0) >= 1.0 - tol_p or g.get("flip_rate", 0) > 0
            checks.append(
                {
                    "alpha": alpha,
                    "check": "positive_steers_impossible",
                    "ok": ok,
                    "got_p": g.get("p_impossible"),
                    "ref_p": r.get("p_impossible"),
                }
            )
        else:
            ok = g.get("p_impossible", 1) <= tol_p or g.get("flip_rate", 0) > 0
            checks.append(
                {
                    "alpha": alpha,
                    "check": "negative_steers_possible",
                    "ok": ok,
                    "got_p": g.get("p_impossible"),
                    "ref_p": r.get("p_impossible"),
                }
            )
        if "flip_rate" in g and "flip_rate" in r:
            checks.append(
                {
                    "alpha": alpha,
                    "check": "flip_rate_within_tol",
                    "ok": _near(float(g["flip_rate"]), float(r["flip_rate"]), tol_flip),
                    "got": g["flip_rate"],
                    "ref": r["flip_rate"],
                }
            )
    passed = sum(1 for c in checks if c.get("ok"))
    return {"checks": checks, "passed": passed, "total": len(checks), "all_ok": passed == len(checks) and len(checks) > 0}


def compare_layer_ablation(
    rows: list[dict[str, Any]],
    *,
    tol_dp: float = 0.25,
) -> dict[str, Any]:
    """Tab. 3 — post-PEZ layers should have zero flip rate."""
    post_pez_zero = all(
        float(r.get("flip_rate", 1)) == 0.0 for r in rows if int(r["injection_layer"]) >= 6
    )
    l5 = next((r for r in rows if int(r["injection_layer"]) == 5), None)
    ref_l5 = next(r for r in table_layer_ablation() if r["injection_layer"] == 5)
    dp_ok = True
    if l5 and ref_l5:
        dp_ok = _near(float(l5["directional_purity"]), float(ref_l5["directional_purity"]), tol_dp) or float(
            l5["directional_purity"]
        ) >= 0.85
    return {
        "post_pez_flip_zero": post_pez_zero,
        "layer5_directional_purity_ok": dp_ok,
        "layer5_dp": float(l5["directional_purity"]) if l5 else None,
        "all_ok": post_pez_zero and dp_ok,
    }


def compare_experiment_to_paper(experiment: dict[str, Any]) -> dict[str, Any]:
    """Summarize how a :func:`run_full_experiment` result aligns with paper claims."""
    alpha = experiment.get("alpha_sweep_test") or experiment.get("alpha_sweep") or []
    ablation = experiment.get("layer_ablation_test") or experiment.get("layer_ablation") or []
    alpha_report = compare_alpha_sweep(alpha)
    ablation_report = compare_layer_ablation(ablation)
    kfold = experiment.get("kfold_accuracy_primary", {})
    ref_layer5 = next(r for r in table_probe_accuracy_by_layer() if r["layer"] == 5)
    probe_ok = float(kfold.get("mean", experiment.get("train_probe_accuracy_primary", 0))) >= 0.55
    block = experiment.get("block_cav", {})
    ref_block = table_block_cav_disentanglement()
    block_angles = block.get("cav_angles_deg", {}) if isinstance(block, dict) else {}
    o2_o3 = block_angles.get("O2_vs_O3")
    block_ok = o2_o3 is None or float(o2_o3) >= 70.0

    sections = {
        "alpha_sweep": alpha_report,
        "layer_ablation": ablation_report,
        "probe_primary": {
            "ok": probe_ok,
            "kfold_mean": kfold.get("mean"),
            "ref_layer5_acc": ref_layer5["val_acc"],
        },
        "block_cav": {"ok": block_ok, "O2_vs_O3_deg": o2_o3, "ref_O2_vs_O3": ref_block["cav_angles_deg"].get("O2_vs_O3")},
    }
    all_ok = (
        alpha_report.get("all_ok", False)
        and ablation_report.get("all_ok", False)
        and probe_ok
        and block_ok
    )
    return {"sections": sections, "all_ok": all_ok, "paper": "arXiv:2605.24322"}


def format_report_markdown(report: dict[str, Any]) -> str:
    lines = ["# Physics steering paper alignment", "", f"Overall: **{'PASS' if report.get('all_ok') else 'REVIEW'}**", ""]
    for name, sec in report.get("sections", {}).items():
        lines.append(f"## {name}")
        if isinstance(sec, dict):
            for k, v in sec.items():
                lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)
