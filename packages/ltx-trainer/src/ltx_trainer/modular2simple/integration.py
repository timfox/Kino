"""GOPEX integration: CARLA, AVstack, driving sim pipelines."""

from __future__ import annotations

from typing import Any


def gopex_stub_links() -> dict[str, str]:
    return {
        "avstack_msma": "AVstack+CARLA terabyte MS/MA dataset generation (arXiv:2606.04444)",
        "anyscene": "Controllable driving scene generation — scenario library source",
        "pinns": "Pedestrian–vehicle interaction benchmark — OpenSCENARIO export target",
        "control_room_17": "Teleplay assets — prop placement validation scenarios",
        "opencs2_render": "CS2 POV renders — orthogonal sim scenario style",
    }


def ltx_plan_stub() -> dict[str, Any]:
    return {
        "phase": "modular_openscenario_atlas",
        "pipeline": [
            "simple_xosc_behavior_templates",
            "cxm_pack_modular_mosc",
            "parameter_reference_customization",
            "cmx_flatten_for_scenario_runner",
            "carla_0915_scenario_runner_execute",
            "scenario_library_loc_audit",
        ],
        "representation": "OpenSCENARIO .xosc + Modular2Simple .mosc packages",
        "prior": "reuse simple scenarios via ScenarioReference + parameterAttributes",
        "scope": "ADS validation scenarios; any OpenSCENARIO-compatible simulator",
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Author simple behaviors with parameterAttributes defaults in ManeuverGroup.",
        "Package with -cxm; customize via ScenarioReference ParameterReference in main.xosc.",
        "Flatten with -cmx before CARLA ScenarioRunner if single-file required.",
        "Set SCENARIO_PATH for shared simple scenarios outside .mosc package.",
        "Expect partial OpenSCENARIO tag support in CARLA — validate maneuvers in sim.",
        "Reuse 2+ behavior templates across library for ~49% LoC reduction (Table 1).",
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_stubs": gopex_stub_links(),
        "ltx_plan": ltx_plan_stub(),
        "prep_notes": ltx_prep_notes(),
    }
