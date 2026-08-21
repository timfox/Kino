"""Modular2Simple configuration (Khriapov et al., OpenSCENARIO modular scenarios)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "modular2simple"
PAPER_TITLE = "Modular2Simple: A Tool for Modular Scenario Creation Based on the OpenSCENARIO Format"
PAPER_AUTHORS = "Nikolai Khriapov, Mohamed Taha Drif, Renjue Li, Cas Widdershoven"
PAPER_URL = "https://github.com/NikolaiKhriapov/modular2simple"
PAPER_REPO = "https://github.com/NikolaiKhriapov/modular2simple"
PAPER_VENUE = "Open-source tool (OpenSCENARIO / CARLA ScenarioRunner)"

CARLA_VERSION = "0.9.15"
SCENARIO_PATH_ENV = "SCENARIO_PATH"

# Case study library (Sec. 4, Table 1)
LIBRARY_SCENARIO_COUNT = 31
LIBRARY_MODULAR_LOC = 10810
LIBRARY_TRADITIONAL_LOC = 21243
LIBRARY_REDUCTION_PCT = 49.1
SIMPLE_REUSE_LOC = (247, 276)


@dataclass
class Modular2SimpleConfig:
    carla_version: str = CARLA_VERSION
    scenario_path_env: str = SCENARIO_PATH_ENV
    library_scenarios: int = LIBRARY_SCENARIO_COUNT
    output_mosc_ext: str = ".mosc"
    output_xosc_ext: str = ".xosc"
    main_file: str = "main.xosc"
