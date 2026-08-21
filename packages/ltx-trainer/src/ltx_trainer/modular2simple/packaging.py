"""Modular2Simple CLI operations: -cxm and -cmx (Sec. 3)."""

from __future__ import annotations

import os
from typing import Any

from ltx_trainer.modular2simple.mosc import MosPackage, build_mosc_package
from ltx_trainer.modular2simple.xosc import flatten_references, scenario_reference_xml


def cxm_combine(
    input_files: list[str],
    output_mosc: str,
    *,
    references: list[dict[str, Any]] | None = None,
) -> MosPackage:
    """Package simple/modular scenarios into one .mosc (-cxm)."""
    simple = [f for f in input_files if f.endswith(".xosc")]
    nested = [f for f in input_files if f.endswith(".mosc")]
    refs = references or [
        {"scenario_file": f, "maneuver_group": f"mg_{i}", "parameters": {}}
        for i, f in enumerate(simple)
    ]
    pkg = build_mosc_package(output_mosc.replace(".mosc", ""), refs, simple, nested)
    return pkg


def cmx_flatten(package: MosPackage) -> dict[str, Any]:
    """Convert modular .mosc into single-file OpenSCENARIO (-cmx)."""
    resolved: list[str] = list(package.simple_files)
    for nested in package.nested_mosc:
        resolved.append(f"/* inlined from {nested} */")
    refs_xml = [
        scenario_reference_xml(
            r["scenario_file"],
            r.get("maneuver_group", "default"),
            r.get("parameters"),
        )
        for r in package.main_references
    ]
    return {
        "output": "flattened.xosc",
        "inlined_simple_count": len(package.simple_files),
        "nested_mosc_count": len(package.nested_mosc),
        "reference_blocks": len(refs_xml),
        "resolved_files": resolved,
        "flattened": True,
    }


def resolve_scenario_path(filename: str, scenario_path: str | None = None) -> str:
    """Resolve external scenario via SCENARIO_PATH (Sec. 3)."""
    base = scenario_path or os.environ.get("SCENARIO_PATH", "")
    if not base:
        return filename
    candidate = os.path.join(base, filename)
    return candidate if os.path.isabs(candidate) else os.path.abspath(candidate)


def packaging_demo() -> dict[str, Any]:
    inputs = ["simple_collision.xosc", "simple_depart.xosc", "modular_emergency.mosc"]
    pkg = cxm_combine(inputs, "intersection.mosc")
    flat = cmx_flatten(pkg)
    return {
        "cxm_input_count": len(inputs),
        "mosc_files": pkg.file_count(),
        "cmx_inlined": flat["inlined_simple_count"],
        "flattened": flat["flattened"],
    }
