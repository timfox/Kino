"""Intersection collision case study (Sec. 4, Fig. 2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.modular2simple.mosc import build_mosc_package


def intersection_case_study_graph() -> dict[str, Any]:
    """Hierarchical modular scenario: collision + emergency response."""
    simple_1 = "simple_1_collision.xosc"
    simple_2 = "simple_2_emergency_arrive.xosc"
    simple_3 = "simple_3_depart.xosc"  # shared via SCENARIO_PATH

    modular_1 = build_mosc_package(
        "modular_1",
        references=[
            {"scenario_file": simple_1, "maneuver_group": "collision", "parameters": {"role": "accident_vehicle"}},
            {"scenario_file": simple_3, "maneuver_group": "depart", "parameters": {"role": "accident_vehicle"}},
        ],
        simple_files=[simple_1, simple_3],
    )
    modular_2 = build_mosc_package(
        "modular_2",
        references=[
            {"scenario_file": simple_2, "maneuver_group": "arrive", "parameters": {"vehicle": "emergency"}},
            {"scenario_file": simple_3, "maneuver_group": "depart", "parameters": {"vehicle": "emergency"}},
        ],
        simple_files=[simple_2, simple_3],
    )
    top = build_mosc_package(
        "intersection_accident",
        references=[
            {"scenario_file": "modular_1.mosc", "maneuver_group": "vehicle_a", "parameters": {}},
            {"scenario_file": "modular_1.mosc", "maneuver_group": "vehicle_b", "parameters": {"lane": "2"}},
            {"scenario_file": "modular_2.mosc", "maneuver_group": "police", "parameters": {}},
            {"scenario_file": "modular_2.mosc", "maneuver_group": "ambulance", "parameters": {}},
        ],
        simple_files=[],
        nested_mosc=["modular_1.mosc", "modular_2.mosc"],
    )
    return {
        "top": top.to_dict(),
        "modular_1": modular_1.to_dict(),
        "modular_2": modular_2.to_dict(),
        "shared_simple": simple_3,
        "actors": ["vehicle_a", "vehicle_b", "police", "ambulance"],
        "phases": ["collision", "emergency_arrival", "departure"],
    }


def case_study_demo() -> dict[str, Any]:
    g = intersection_case_study_graph()
    return {
        "scenario_name": g["top"]["name"],
        "nested_modular_count": len(g["top"]["nested_mosc"]),
        "shared_simple_reuse": g["shared_simple"],
        "actor_count": len(g["actors"]),
    }
