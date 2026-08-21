"""Built-in survey catalog (Table 1, v2.5.0)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.survey_footprint.config import FULL_SKY_DEG2


@dataclass(frozen=True)
class SurveyRecord:
    survey_id: str
    label: str
    wavelength: str
    area_deg2: float
    reference: str

    @property
    def sky_fraction(self) -> float:
        return self.area_deg2 / FULL_SKY_DEG2


# Table 1 — areas from distributed FITS MOC products (paper v2.5.0)
BUILTIN_SURVEYS: tuple[SurveyRecord, ...] = (
    SurveyRecord("kids", "KiDS", "opt", 891.2, "Jong et al. (2013)"),
    SurveyRecord("euclid_dr1", "Euclid DR1", "opt/NIR", 2108.5, "Euclid Collaboration et al. (2025)"),
    SurveyRecord("hsc", "HSC", "opt/NIR", 1653.4, "Aihara et al. (2018)"),
    SurveyRecord("des", "DES", "opt/NIR", 5155.0, "DES Collaboration et al. (2016)"),
    SurveyRecord("unions", "UNIONS", "opt", 6194.2, "Gwyn et al. (2025)"),
    SurveyRecord("desi_legacy_dr9", "DESI Legacy DR9", "opt/IR", 20813.1, "Dey et al. (2019)"),
    SurveyRecord("erass1", "eRASS1", "X-ray", 21524.4, "Merloni et al. (2024)"),
    SurveyRecord("lsst_wfd", "LSST WFD", "opt", 17719.2, "Ivezić et al. (2019)"),
    SurveyRecord("roman_hlwas", "Roman HLWAS", "NIR", 5314.0, "Akeson et al. (2019)"),
    SurveyRecord("roman_hlwas_deep", "Roman HLWAS Deep", "opt/NIR", 35.6, "Hirata (2024)"),
    SurveyRecord("roman_hltds", "Roman HLTDS", "opt/NIR", 28.1, "Roman OTAC (2025)"),
    SurveyRecord("roman_hltds_deep", "Roman HLTDS Deep", "opt/NIR", 7.7, "Roman OTAC (2025)"),
    SurveyRecord("act_legacy", "ACT Legacy", "mm", 11245.0, "Aguena et al. (2026)"),
)

SURVEY_BY_ID: dict[str, SurveyRecord] = {s.survey_id: s for s in BUILTIN_SURVEYS}
