"""Survey Footprint Explorer configuration (Ahad et al. arXiv:2605.11099)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2605.11099"
PAPER_TITLE = (
    "Survey Footprint Explorer: A Browser-Based Interactive Tool for "
    "Visualizing and Cross-Matching Astronomical Survey Footprints"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
TOOL_URL = "https://www.lammimahad.com/survey-footprint-explorer"
INSTRUCTIONS_URL = "https://www.lammimahad.com/survey-explorer-instructions"
VERSION = "2.5.0"

# Full-sky area (deg²) used with MOC sky_fraction (Sec. 3.1)
FULL_SKY_DEG2 = 41_252.96

COLOR_THEMES = ("rainbow", "iridescent", "vivid")
UI_THEMES = ("dark", "light")

# Use-case surveys (Sec. 5.1–5.2)
USE_CASE_SURVEYS = ("euclid_dr1", "lsst_wfd", "roman_hlwas")


@dataclass
class SurveyFootprintConfig:
    mc_samples: int = 20_000
    default_theme: str = "rainbow"
    remember_session: bool = True
