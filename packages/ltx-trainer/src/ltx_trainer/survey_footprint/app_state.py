"""Browser session state model (Sec. 2.9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    selected_surveys: list[str] = field(default_factory=list)
    survey_order: list[str] = field(default_factory=list)
    color_theme: str = "rainbow"
    ui_theme: str = "dark"
    projection: str = "aladin"
    cross_match_only: bool = False
    show_galactic: bool = False
    show_ecliptic: bool = False
    remember: bool = True
    catalogue_rows: list[dict[str, Any]] | None = None
    custom_moc_id: str | None = None

    def to_local_storage(self) -> dict[str, Any]:
        return {
            "selected_surveys": self.selected_surveys,
            "survey_order": self.survey_order,
            "color_theme": self.color_theme,
            "ui_theme": self.ui_theme,
            "projection": self.projection,
            "cross_match_only": self.cross_match_only,
            "show_galactic": self.show_galactic,
            "show_ecliptic": self.show_ecliptic,
            "remember": self.remember,
            "custom_moc_id": self.custom_moc_id,
        }

    @classmethod
    def from_local_storage(cls, data: dict[str, Any]) -> AppState:
        return cls(
            selected_surveys=list(data.get("selected_surveys", [])),
            survey_order=list(data.get("survey_order", [])),
            color_theme=str(data.get("color_theme", "rainbow")),
            ui_theme=str(data.get("ui_theme", "dark")),
            projection=str(data.get("projection", "aladin")),
            cross_match_only=bool(data.get("cross_match_only", False)),
            show_galactic=bool(data.get("show_galactic", False)),
            show_ecliptic=bool(data.get("show_ecliptic", False)),
            remember=bool(data.get("remember", True)),
            custom_moc_id=data.get("custom_moc_id"),
        )
