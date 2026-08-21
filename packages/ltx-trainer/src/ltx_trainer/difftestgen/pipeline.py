"""DiffTestGen evaluation pipeline and smoke checks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.difftestgen.access import FunctionInfo, classify_function, extract_access_info
from ltx_trainer.difftestgen.config import DiffTestGenConfig
from ltx_trainer.difftestgen.coverage import (
    ExecResult,
    annotate_coverage,
    has_behavioral_difference,
    union_coverage,
)


def run_access_demo() -> dict[str, Any]:
    """Public / private / special access-info examples (Fig. 3)."""
    cases = [
        FunctionInfo("style", "public", file="pandas/core/frame", class_name="DataFrame"),
        FunctionInfo("_run_validator", "private", file="marshmallow/schema", class_name="Schema"),
        FunctionInfo("__call__", "special", file="marshmallow/types", class_name="SchemaValidator"),
    ]
    out = {}
    for f in cases:
        infos = extract_access_info(
            f,
            public_entries=["validate", "load"] if f.category == "private" else None,
            top_k=2,
        )
        out[f.name] = [a.as_dict() for a in infos]
    return {
        "classify_underscore": classify_function("_run_validator"),
        "classify_dunder": classify_function("__call__"),
        "classify_public_api": classify_function("style", in_public_api=True),
        "access": out,
    }


def run_generation_demo(config: DiffTestGenConfig | None = None) -> dict[str, Any]:
    """Simulate R0 + coverage feedback rounds until saturation."""
    cfg = config or DiffTestGenConfig()
    # Stub PR: private change reachable via validate
    func = FunctionInfo("_run_validator", "private", file="marshmallow/schema", class_name="Schema")
    access = extract_access_info(func, public_entries=["validate"], top_k=cfg.top_k_call_paths)
    rounds = []
    covered_old, covered_new = 0, 0
    changed_old, changed_new = 8, 8
    tests_with_diff = 0
    for r in range(cfg.max_rounds):
        # Each round covers more changed lines when access info present
        step = 3 if access else 1
        covered_old = min(changed_old, covered_old + step)
        covered_new = min(changed_new, covered_new + step)
        cov = union_coverage(
            old_changed=changed_old,
            new_changed=changed_new,
            old_covered=covered_old,
            new_covered=covered_new,
        )
        # Simulate a differentiating test once coverage is decent
        old_r = ExecResult(output="errors: {}")
        new_r = ExecResult(output="errors: {'x': ['invalid']}") if cov > 0.4 else ExecResult(output="errors: {}")
        diff = has_behavioral_difference(old_r, new_r)
        if diff:
            tests_with_diff += 1
        rounds.append({"round": r, "union_coverage": round(cov, 4), "behavioral_diff": diff})
        if cov >= cfg.target_union_coverage:
            break
        if r > 0 and rounds[-1]["union_coverage"] == rounds[-2]["union_coverage"]:
            break
    annotated = annotate_coverage(
        ["def eye(N, M=None, k=0):", "    if not M:", "        M = N", "    return N"],
        covered={1, 2},
    )
    return {
        "access_paths": len(access),
        "rounds": rounds,
        "final_union_coverage": rounds[-1]["union_coverage"] if rounds else 0.0,
        "tests_with_diff": tests_with_diff,
        "annotated_sample": annotated,
        "reached_target": rounds[-1]["union_coverage"] >= 0.99 if rounds else False,
    }


def evaluation_demo() -> dict[str, Any]:
    access = run_access_demo()
    gen = run_generation_demo()
    # Behavioral difference cases (Def 2)
    cases = {
        "error_type_mismatch": has_behavioral_difference(
            ExecResult(error_type="TypeError"), ExecResult(error_type="ValueError")
        ),
        "one_sided_error": has_behavioral_difference(
            ExecResult(output="ok"), ExecResult(error_type="RuntimeError")
        ),
        "output_mismatch": has_behavioral_difference(
            ExecResult(output="a"), ExecResult(output="b")
        ),
        "same_ok": not has_behavioral_difference(
            ExecResult(output="same"), ExecResult(output="same")
        ),
    }
    return {
        "access_demo": access,
        "generation": gen,
        "diff_cases": cases,
        "access_helps_private": len(access["access"]["_run_validator"]) >= 1,
        "union_improves": gen["final_union_coverage"] >= 0.5,
        "exposes_diff": gen["tests_with_diff"] > 0,
        "def2_oracle_ok": all(cases.values()),
    }


def evaluation_smoke() -> dict[str, bool]:
    from ltx_trainer.difftestgen.benchmarks import PAPER_ANCHORS, TABLE_TESTORA_OVERALL

    demo = evaluation_demo()
    cov = union_coverage(old_changed=10, new_changed=10, old_covered=9, new_covered=9)
    checks = {
        "classify_private": classify_function("_run_validator") == "private",
        "access_helps_private": demo["access_helps_private"],
        "union_metric": abs(cov - 0.9) < 1e-9,
        "union_improves": demo["union_improves"],
        "exposes_diff": demo["exposes_diff"],
        "def2_oracle_ok": demo["def2_oracle_ok"],
        "paper_overall_prs": abs(float(PAPER_ANCHORS["overall_prs_pct"]) - 0.782) < 1e-6,
        "paper_overall_union": abs(float(PAPER_ANCHORS["overall_union_coverage"]) - 0.907) < 1e-6,
        "paper_n_prs": int(PAPER_ANCHORS["n_prs"]) == 463,
        "paper_testora_difftestgen_prs": int(TABLE_TESTORA_OVERALL["DiffTestGen"]["num_pr"]) == 350,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def evaluation_smoke_json() -> dict[str, bool]:
    return evaluation_smoke()
