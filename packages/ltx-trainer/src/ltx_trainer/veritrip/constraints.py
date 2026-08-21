"""Commonsense and preference evaluators (Table 7, Sec. 3.4)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ltx_trainer.veritrip.config import VeriTripConfig, VeriTripQuery
from ltx_trainer.veritrip.matching import fuzzy_match, parse_time_hhmm
from ltx_trainer.veritrip.schema import TravelPlan


def _date_range(start: str, end: str) -> set[str]:
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    out: set[str] = set()
    cur = d0
    while cur <= d1:
        out.add(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _room_capacity(room_type: str) -> int:
    rt = room_type.lower()
    if "family" in rt:
        return 4
    if "twin" in rt or "double" in rt:
        return 2
    if "single" in rt:
        return 1
    return 2


def evaluate_constraints(
    plan: TravelPlan,
    query: VeriTripQuery,
    *,
    cfg: VeriTripConfig | None = None,
) -> dict[str, bool]:
    """Per-constraint boolean results (Table 7)."""
    cfg = cfg or VeriTripConfig()
    results: dict[str, bool] = {}

    total_cost = sum(t.price_per_person * query.people_number for t in plan.transportation)
    total_cost += sum(a.price_per_night * a.room_number for a in plan.accommodation)
    total_cost += sum(i.price_per_person * query.people_number for i in plan.itinerary)

    if query.budget is not None:
        cap = query.budget * (1.0 + cfg.budget_tolerance)
        results["is_within_budget"] = total_cost <= cap
    else:
        results["is_within_budget"] = True

    # Transportation timing & continuity
    if plan.transportation:
        first = plan.transportation[0]
        last = plan.transportation[-1]
        results["transportation_dates"] = (
            first.date == query.start_date and last.date == query.end_date
        )
        results["transportation_closed_loop"] = (
            fuzzy_match(first.departure_station, last.arrive_station)
            and fuzzy_match(last.departure_station, first.arrive_station)
        )
        continuity = True
        for i in range(len(plan.transportation) - 1):
            if not fuzzy_match(plan.transportation[i].arrive_station, plan.transportation[i + 1].departure_station):
                continuity = False
        results["transportation_continuity"] = continuity

        arrive_m = parse_time_hhmm(first.end_time)
        depart_m = parse_time_hhmm(last.begin_time)
        non_transport = [i for i in plan.itinerary if i.active_type != "inter_city_transportation"]
        if non_transport and arrive_m is not None:
            first_act = min(non_transport, key=lambda x: parse_time_hhmm(x.start_time) or 0)
            results["first_activity_after_arrival"] = (parse_time_hhmm(first_act.start_time) or 0) >= arrive_m
        else:
            results["first_activity_after_arrival"] = True
        if non_transport and depart_m is not None:
            last_act = max(non_transport, key=lambda x: parse_time_hhmm(x.end_time) or 0)
            results["last_activity_before_departure"] = (parse_time_hhmm(last_act.end_time) or 9999) <= depart_m
        else:
            results["last_activity_before_departure"] = True
    else:
        for k in (
            "transportation_dates",
            "transportation_closed_loop",
            "transportation_continuity",
            "first_activity_after_arrival",
            "last_activity_before_departure",
        ):
            results[k] = False

    # Accommodation
    if plan.accommodation and query.people_number:
        cap = sum(_room_capacity(a.room_type) * a.room_number for a in plan.accommodation)
        results["accommodation_capacity"] = cap >= query.people_number
        expected = _date_range(query.start_date, query.end_date)
        # nights: check_in .. check_out-1 covers trip
        covered: set[str] = set()
        for a in plan.accommodation:
            d_in = datetime.strptime(a.check_in, "%Y-%m-%d")
            d_out = datetime.strptime(a.check_out, "%Y-%m-%d")
            cur = d_in
            while cur < d_out:
                covered.add(cur.strftime("%Y-%m-%d"))
                cur += timedelta(days=1)
        results["accommodation_coverage"] = expected.issubset(covered) or len(covered) >= len(expected) - 1
    else:
        results["accommodation_capacity"] = bool(not query.people_number)
        results["accommodation_coverage"] = False

    # Uniqueness
    attr_names = [i.name for i in plan.itinerary if i.active_type == "attraction"]
    meal_names = [i.name for i in plan.itinerary if i.active_type == "meal"]
    results["attractions_uniqueness"] = len(attr_names) == len(set(attr_names))
    results["meals_uniqueness"] = len(meal_names) == len(set(meal_names))

    # Days coverage
    expected_days = _date_range(query.start_date, query.end_date)
    plan_days = {i.date for i in plan.itinerary}
    results["days_coverage_correct"] = plan_days == expected_days

    mid_days = sorted(expected_days)[1:-1] if len(expected_days) > 2 else []
    mid_ok = True
    for d in mid_days:
        day_items = [i for i in plan.itinerary if i.date == d]
        has_meal = any(i.active_type == "meal" for i in day_items)
        has_attr = any(i.active_type == "attraction" for i in day_items)
        if not (has_meal and has_attr):
            mid_ok = False
    results["activity_presence_on_mid_days"] = mid_ok if mid_days else True

    min_density = max(2, len(expected_days) * 2)
    results["information_density"] = len(plan.itinerary) >= min_density

    # Inner-city time efficiency (simplified)
    results["inner_city_time_efficiency"] = True
    for d in expected_days:
        day_items = [i for i in plan.itinerary if i.date == d]
        commute = sum(
            (parse_time_hhmm(i.end_time) or 0) - (parse_time_hhmm(i.start_time) or 0)
            for i in day_items
            if i.active_type == "inner_city_transportation"
        )
        total = sum(
            (parse_time_hhmm(i.end_time) or 0) - (parse_time_hhmm(i.start_time) or 0)
            for i in day_items
        )
        if total > 0 and commute / total > cfg.inner_city_time_ratio_limit:
            results["inner_city_time_efficiency"] = False

    # Preferences (Table 7)
    prefs = query.preferences
    if prefs.get("p_intercity_traffic"):
        mode = prefs["p_intercity_traffic"].lower()
        ids = " ".join(t.transportation_id for t in plan.transportation).upper()
        if "high" in mode or "rail" in mode:
            results["intercity_transportation_preference"] = "G" in ids or "D" in ids
        elif "air" in mode or "plane" in mode:
            results["intercity_transportation_preference"] = any(c.isalpha() and c.isupper() for c in ids.split())
        else:
            results["intercity_transportation_preference"] = True
    else:
        results["intercity_transportation_preference"] = True

    if prefs.get("p_accommodation"):
        want = [p.strip() for p in prefs["p_accommodation"].split("|") if p.strip()]
        results["accommodation_preference"] = any(
            any(fuzzy_match(w, a.room_type) for w in want) for a in plan.accommodation
        )
    else:
        results["accommodation_preference"] = True

    if prefs.get("p_attraction"):
        must = [p.strip() for p in prefs["p_attraction"].split("|") if p.strip()]
        planned = [i.name for i in plan.itinerary if i.active_type == "attraction"]
        results["attraction_preference"] = all(
            any(fuzzy_match(m, p) for p in planned) for m in must
        )
    else:
        results["attraction_preference"] = True

    if prefs.get("p_meal"):
        want = [p.strip() for p in prefs["p_meal"].split("|") if p.strip()]
        meals = [i.name for i in plan.itinerary if i.active_type == "meal"]
        results["meal_preference"] = any(
            any(fuzzy_match(w, m) for m in meals) for w in want
        )
    else:
        results["meal_preference"] = True

    results["inner_city_transportation_preference"] = True

    return results


HARD_CONSTRAINTS = (
    "is_within_budget",
    "first_activity_after_arrival",
    "last_activity_before_departure",
    "transportation_continuity",
    "transportation_closed_loop",
    "transportation_dates",
    "accommodation_capacity",
    "accommodation_coverage",
    "attractions_uniqueness",
    "meals_uniqueness",
    "inner_city_time_efficiency",
    "days_coverage_correct",
    "activity_presence_on_mid_days",
    "information_density",
)

PREFERENCE_KEYS = (
    "intercity_transportation_preference",
    "accommodation_preference",
    "attraction_preference",
    "meal_preference",
    "inner_city_transportation_preference",
)


def pass_rate_micro(results: dict[str, bool]) -> float:
    keys = [k for k in HARD_CONSTRAINTS if k in results]
    if not keys:
        return 0.0
    return sum(1 for k in keys if results[k]) / len(keys)


def pass_rate_macro(results: dict[str, bool]) -> float:
    return 1.0 if all(results.get(k, False) for k in HARD_CONSTRAINTS) else 0.0


def preference_fulfillment_rate(results: dict[str, bool]) -> float:
    keys = [k for k in PREFERENCE_KEYS if k in results]
    if not keys:
        return 1.0
    return sum(1 for k in keys if results[k]) / len(keys)


def constraint_table() -> list[dict[str, str]]:
    """Table 7 — constraint descriptions for agents."""
    return [
        {"name": "is_within_budget", "description": "Total cost within budget (+10% tolerance)."},
        {"name": "transportation_dates", "description": "First/last leg dates match query window."},
        {"name": "attraction_preference", "description": "Must-see attractions included."},
        {"name": "meal_preference", "description": "Dining matches stated tastes."},
        {"name": "days_coverage_correct", "description": "Activity dates cover full trip."},
    ]
