"""Framework card, evaluation demo, and smoke entry points."""

from __future__ import annotations

from typing import Any

from ltx_trainer.modular2simple.benchmarks import benchmarks_bundle
from ltx_trainer.modular2simple.case_study import case_study_demo
from ltx_trainer.modular2simple.config import Modular2SimpleConfig
from ltx_trainer.modular2simple.integration import integration_bundle
from ltx_trainer.modular2simple.library import library_demo
from ltx_trainer.modular2simple.packaging import packaging_demo
from ltx_trainer.modular2simple.xosc import xosc_demo


def framework_card(cfg: Modular2SimpleConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Modular2SimpleConfig()
    return {
        "paper": benchmarks_bundle()["paper"],
        "method": {
            "tool": "Modular2Simple Java CLI",
            "format": "OpenSCENARIO .xosc + custom .mosc packages",
            "operations": {
                "cxm": "combine simple/modular → .mosc",
                "cmx": "flatten .mosc → single .xosc",
            },
            "extensions": "ScenarioReference, parameterAttributes on ManeuverGroup",
            "simulator": f"CARLA {cfg.carla_version} ScenarioRunner (+ any OpenSCENARIO host)",
        },
        "config": cfg.__dict__,
        "integration": integration_bundle(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    _ = seed
    lib = library_demo()
    return {
        "xosc": xosc_demo(),
        "packaging": packaging_demo(),
        "case_study": case_study_demo(),
        "library": lib,
        "ref_reduction_pct": lib["reduction_pct"],
        "ref_modular_loc": lib["modular_loc"],
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    return {
        "package": "modular2simple",
        "paper": "modular2simple",
        "arxiv": "modular2simple",
        "ref_reduction_pct": demo["ref_reduction_pct"],
        "ref_library_scenarios": 31,
        "ref_modular_loc": demo["ref_modular_loc"],
        "case_study_actors": demo["case_study"]["actor_count"],
        "gopex_stub_count": len(integration_bundle()["gopex_stubs"]),
    }
