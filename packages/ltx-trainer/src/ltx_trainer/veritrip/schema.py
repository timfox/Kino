"""Structured plan JSON schema and parsing (Sec. 3.1, Appendix E)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


REQUIRED_TOP_KEYS = frozenset({"transportationTable", "accommodationTable", "itineraryTable"})


@dataclass
class TransportationLeg:
    transportation_id: str
    date: str
    departure_station: str
    arrive_station: str
    begin_time: str
    end_time: str
    price_per_person: int


@dataclass
class AccommodationRow:
    city: str
    name: str
    check_in: str
    check_out: str
    price_per_night: int
    room_type: str
    room_number: int
    breakfast_included: bool


@dataclass
class ItineraryItem:
    date: str
    active_item_number: str
    active_type: str
    start_time: str
    end_time: str
    name: str
    price_per_person: int
    notes: str


@dataclass
class TravelPlan:
    transportation: list[TransportationLeg] = field(default_factory=list)
    accommodation: list[AccommodationRow] = field(default_factory=list)
    itinerary: list[ItineraryItem] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _unwrap_final_result(obj: dict[str, Any]) -> dict[str, Any]:
    if "Final Result" in obj:
        return obj["Final Result"]
    if "final_result" in obj:
        return obj["final_result"]
    return obj


def parse_travel_plan(agent_output: str | dict[str, Any]) -> tuple[TravelPlan | None, str | None]:
    """
    Format check (Sec. 3.4): strictly parsable JSON with three top-level tables.
    Returns (plan, error).
    """
    try:
        if isinstance(agent_output, dict):
            data = agent_output
        else:
            text = agent_output.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, str(exc)

    if not isinstance(data, dict):
        return None, "root must be object"

    body = _unwrap_final_result(data)
    if not REQUIRED_TOP_KEYS.issubset(body.keys()):
        missing = REQUIRED_TOP_KEYS - set(body.keys())
        return None, f"missing keys: {sorted(missing)}"

    plan = TravelPlan(raw=body)

    for row in body.get("transportationTable") or []:
        if not isinstance(row, dict):
            continue
        plan.transportation.append(
            TransportationLeg(
                transportation_id=str(row.get("transportationID", "")),
                date=str(row.get("date", "")),
                departure_station=str(row.get("departureStation", "")),
                arrive_station=str(row.get("arriveStation", "")),
                begin_time=str(row.get("begin_time", "")),
                end_time=str(row.get("end_time", "")),
                price_per_person=int(row.get("price_per_person", 0) or 0),
            )
        )

    for row in body.get("accommodationTable") or []:
        if not isinstance(row, dict):
            continue
        bf = row.get("breakfast_included", False)
        plan.accommodation.append(
            AccommodationRow(
                city=str(row.get("city", "")),
                name=str(row.get("name", "")),
                check_in=str(row.get("check_in", "")),
                check_out=str(row.get("check_out", "")),
                price_per_night=int(row.get("price_per_night", 0) or 0),
                room_type=str(row.get("room_type", "")),
                room_number=int(row.get("room_number", 1) or 1),
                breakfast_included=bf in (True, "true", "True", 1, "1"),
            )
        )

    for row in body.get("itineraryTable") or []:
        if not isinstance(row, dict):
            continue
        plan.itinerary.append(
            ItineraryItem(
                date=str(row.get("Date", row.get("date", ""))),
                active_item_number=str(row.get("active_item_number", "")),
                active_type=str(row.get("active_type", "")),
                start_time=str(row.get("start_time", "")),
                end_time=str(row.get("end_time", "")),
                name=str(row.get("name", "")),
                price_per_person=int(row.get("price_per_person", 0) or 0),
                notes=str(row.get("notes", "")),
            )
        )

    return plan, None


def delivery_rate(outputs: list[str | dict[str, Any]]) -> float:
    """DR ↑ — fraction of strictly valid JSON plans."""
    if not outputs:
        return 0.0
    ok = sum(1 for o in outputs if parse_travel_plan(o)[0] is not None)
    return ok / len(outputs)
