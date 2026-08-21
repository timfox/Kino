"""OpenSCENARIO XML fragments (ScenarioReference, parameterAttributes)."""

from __future__ import annotations

from typing import Any
from xml.etree import ElementTree as ET


def scenario_reference_xml(
    scenario_file: str,
    maneuver_group: str,
    parameters: dict[str, str] | None = None,
) -> str:
    """Build ScenarioReference block (Sec. 3)."""
    root = ET.Element("ScenarioReference", scenarioFileName=scenario_file)
    mg = ET.SubElement(root, "ManeuverGroupReference", maneuverGroupName=maneuver_group)
    for key, value in (parameters or {}).items():
        ET.SubElement(mg, "ParameterReference", key=key, value=value)
    return ET.tostring(root, encoding="unicode")


def simple_maneuver_group_xml(
    name: str,
    parameter_defaults: dict[str, str],
) -> str:
    """ManeuverGroup with parameterAttributes for reusable simple scenarios."""
    attrs = " ".join(f'{k}="{v}"' for k, v in parameter_defaults.items())
    param_keys = " ".join(parameter_defaults.keys())
    root = ET.Element(
        "ManeuverGroup",
        name=name,
        parameterAttributes=param_keys,
    )
    cond = ET.SubElement(root, "Condition")
    cond.set("parameterAttributes", param_keys)
    for k, v in parameter_defaults.items():
        cond.set(k, v)
    ET.SubElement(cond, "ByValue").text = "placeholder"
    return ET.tostring(root, encoding="unicode")


def flatten_references(references: list[dict[str, Any]]) -> list[str]:
    """Collect referenced scenario filenames in declaration order."""
    return [r["scenario_file"] for r in references]


def xosc_demo() -> dict[str, Any]:
    ref = scenario_reference_xml(
        "simple_collision.xosc",
        "collision_maneuver",
        {"speed": "30", "lane": "1"},
    )
    return {
        "has_scenario_reference": "ScenarioReference" in ref,
        "has_parameter_reference": "ParameterReference" in ref,
        "reference_count": 1,
    }
