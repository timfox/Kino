"""Verifiable Knowledge Base — agent-inaccessible ground truth (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.veritrip.matching import exact_match, fuzzy_match


@dataclass
class TransportationRecord:
    transportation_id: str
    date: str
    begin_time: str
    end_time: str
    departure_station: str
    arrive_station: str
    price_per_person: int


@dataclass
class EntityRecord:
    name: str
    city: str
    lat: float = 0.0
    lon: float = 0.0
    opening_hours: str = ""
    price: int = 0


@dataclass
class VerifiableKnowledgeBase:
    """Structured facts extracted from MRB; used only by the evaluator."""

    transportation: list[TransportationRecord] = field(default_factory=list)
    attractions: list[EntityRecord] = field(default_factory=list)
    accommodations: list[EntityRecord] = field(default_factory=list)
    restaurants: list[EntityRecord] = field(default_factory=list)
    pois: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (lat, lon)

    def find_transport(self, transport_id: str, date: str) -> TransportationRecord | None:
        for t in self.transportation:
            if exact_match(t.transportation_id, transport_id) and exact_match(t.date, date):
                return t
        return None

    def match_attraction(self, name: str, city: str, *, threshold: float = 0.85) -> bool:
        for a in self.attractions:
            if fuzzy_match(a.name, name, threshold=threshold) and (
                not city or fuzzy_match(a.city, city, threshold=threshold)
            ):
                return True
        return False

    def match_accommodation(self, name: str, city: str, *, threshold: float = 0.85) -> bool:
        for h in self.accommodations:
            if fuzzy_match(h.name, name, threshold=threshold) and (
                not city or fuzzy_match(h.city, city, threshold=threshold)
            ):
                return True
        return False

    def poi_coordinates(self, name: str, *, threshold: float = 0.85) -> tuple[float, float] | None:
        for pname, coord in self.pois.items():
            if fuzzy_match(pname, name, threshold=threshold):
                return coord
        for a in self.attractions:
            if fuzzy_match(a.name, name, threshold=threshold):
                return (a.lat, a.lon)
        return None

    def stats(self) -> dict[str, int]:
        return {
            "transportation": len(self.transportation),
            "restaurants": len(self.restaurants),
            "accommodations": len(self.accommodations),
            "attractions": len(self.attractions),
            "pois": len(self.pois),
        }


def build_demo_vkb() -> VerifiableKnowledgeBase:
    """Minimal VKB for Changsha demo (Figure 4 bronze artifact / museum case)."""
    return VerifiableKnowledgeBase(
        transportation=[
            TransportationRecord(
                transportation_id="G1011",
                date="2025-10-28",
                begin_time="08:00",
                end_time="14:30",
                departure_station="Guangzhou South",
                arrive_station="Changsha South",
                price_per_person=553,
            ),
            TransportationRecord(
                transportation_id="G1012",
                date="2025-10-30",
                begin_time="15:00",
                end_time="21:00",
                departure_station="Changsha South",
                arrive_station="Guangzhou South",
                price_per_person=553,
            ),
        ],
        attractions=[
            EntityRecord("Hunan Provincial Museum", "Changsha", 28.211, 112.991),
            EntityRecord("Orange Isle", "Changsha", 28.189, 112.960),
            EntityRecord("Yuelu Academy", "Changsha", 28.179, 112.936),
        ],
        accommodations=[
            EntityRecord("Changsha W Hotel", "Changsha", 28.196, 113.003),
        ],
        restaurants=[
            EntityRecord("Old Hunan Tea House", "Changsha", 28.195, 112.982),
            EntityRecord("Fire Palace", "Changsha", 28.192, 112.978),
        ],
        pois={
            "Hunan Provincial Museum": (28.211, 112.991),
            "Orange Isle": (28.189, 112.960),
            "Yuelu Academy": (28.179, 112.936),
        },
    )
