"""VeriTrip evaluation metrics: DR, FR, PR, PFR, AM (Sec. 3.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.veritrip.config import VeriTripConfig, VeriTripQuery
from ltx_trainer.veritrip.constraints import (
    evaluate_constraints,
    pass_rate_macro,
    pass_rate_micro,
    preference_fulfillment_rate,
)
from ltx_trainer.veritrip.geography import average_margin, route_length_km, tsp_optimal_km
from ltx_trainer.veritrip.matching import exact_match, fuzzy_match
from ltx_trainer.veritrip.schema import TravelPlan, parse_travel_plan
from ltx_trainer.veritrip.vkb import VerifiableKnowledgeBase


@dataclass
class FactCell:
    category: str
    field: str
    value: str
    correct: bool


@dataclass
class EvaluationScores:
    delivered: bool
    delivery_rate: float
    factual_reliability: float
    pass_rate_micro: float
    pass_rate_macro: float
    preference_fulfillment_rate: float
    average_margin: float
    fact_cells: list[FactCell] = field(default_factory=list)
    constraint_results: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivered": self.delivered,
            "DR": round(self.delivery_rate, 2),
            "FR": round(self.factual_reliability, 2),
            "PRmi": round(self.pass_rate_micro, 2),
            "PRma": round(self.pass_rate_macro, 2),
            "PFR": round(self.preference_fulfillment_rate, 2),
            "AM": round(self.average_margin, 2),
            "fact_cells_total": len(self.fact_cells),
            "fact_cells_correct": sum(1 for c in self.fact_cells if c.correct),
        }


def cell_wise_fact_check(
    plan: TravelPlan,
    vkb: VerifiableKnowledgeBase,
    *,
    cfg: VeriTripConfig | None = None,
) -> list[FactCell]:
    """Cell-wise FR against VKB (Sec. 3.4, Table 3)."""
    cfg = cfg or VeriTripConfig()
    cells: list[FactCell] = []

    for leg in plan.transportation:
        rec = vkb.find_transport(leg.transportation_id, leg.date)
        cells.append(
            FactCell("transportation", "transportationID", leg.transportation_id, rec is not None)
        )
        if rec:
            cells.append(
                FactCell("transportation", "begin_time", leg.begin_time, exact_match(leg.begin_time, rec.begin_time))
            )
            cells.append(
                FactCell("transportation", "end_time", leg.end_time, exact_match(leg.end_time, rec.end_time))
            )
        else:
            cells.append(FactCell("transportation", "begin_time", leg.begin_time, False))
            cells.append(FactCell("transportation", "end_time", leg.end_time, False))

    for item in plan.itinerary:
        if item.active_type == "attraction":
            ok = vkb.match_attraction(item.name, "", threshold=cfg.fuzzy_match_threshold)
            cells.append(FactCell("attraction", "name", item.name, ok))
        elif item.active_type == "meal":
            # Restaurants: preference validation only in paper; optional name check
            ok = any(
                fuzzy_match(item.name, r.name, threshold=cfg.fuzzy_match_threshold)
                for r in vkb.restaurants
            ) or bool(item.name)
            cells.append(FactCell("meal", "name", item.name, ok))

    for acc in plan.accommodation:
        ok = vkb.match_accommodation(acc.name, acc.city, threshold=cfg.fuzzy_match_threshold)
        cells.append(FactCell("accommodation", "name", acc.name, ok))

    return cells


def factual_reliability(cells: list[FactCell]) -> float:
    if not cells:
        return 0.0
    return sum(1 for c in cells if c.correct) / len(cells)


def geographic_average_margin(
    plan: TravelPlan,
    vkb: VerifiableKnowledgeBase,
    *,
    cfg: VeriTripConfig | None = None,
) -> float:
    cfg = cfg or VeriTripConfig()
    coords: list[tuple[float, float]] = []
    for item in plan.itinerary:
        if item.active_type == "attraction":
            c = vkb.poi_coordinates(item.name, threshold=cfg.fuzzy_match_threshold)
            if c:
                coords.append(c)
    if len(coords) < 2:
        return 0.0
    agent_km = route_length_km(coords)
    ref_km = tsp_optimal_km(coords)
    return average_margin(agent_km, ref_km, len(coords), scale_10km=cfg.am_scale_10km)


def evaluate_plan(
    agent_output: str | dict[str, Any],
    query: VeriTripQuery,
    vkb: VerifiableKnowledgeBase,
    *,
    cfg: VeriTripConfig | None = None,
) -> EvaluationScores:
    """Full programmatic pipeline (Sec. 3.4)."""
    cfg = cfg or VeriTripConfig()
    plan, err = parse_travel_plan(agent_output)
    if plan is None:
        return EvaluationScores(
            delivered=False,
            delivery_rate=0.0,
            factual_reliability=0.0,
            pass_rate_micro=0.0,
            pass_rate_macro=0.0,
            preference_fulfillment_rate=0.0,
            average_margin=0.0,
        )

    cells = cell_wise_fact_check(plan, vkb, cfg=cfg)
    fr = factual_reliability(cells)
    constraints = evaluate_constraints(plan, query, cfg=cfg)
    return EvaluationScores(
        delivered=True,
        delivery_rate=1.0,
        factual_reliability=fr,
        pass_rate_micro=pass_rate_micro(constraints),
        pass_rate_macro=pass_rate_macro(constraints),
        preference_fulfillment_rate=preference_fulfillment_rate(constraints),
        average_margin=geographic_average_margin(plan, vkb, cfg=cfg),
        fact_cells=cells,
        constraint_results=constraints,
    )
