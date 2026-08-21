"""VeriTrip framework card, demo environment, evaluation smoke."""

from __future__ import annotations

import json
from typing import Any

from ltx_trainer.veritrip.agent_tools import build_demo_toolset
from ltx_trainer.veritrip.config import AGENT_TOOLS, VeriTripConfig, VeriTripQuery
from ltx_trainer.veritrip.constraints import constraint_table
from ltx_trainer.veritrip.metrics import evaluate_plan
from ltx_trainer.veritrip.tables import (
    table2_dataset_statistics,
    table3_evaluation_mapping,
    table4_main_results,
    table5_visual_ablation,
    table6_noisy_mrb,
)
from ltx_trainer.veritrip.vkb import build_demo_vkb


def framework_card(cfg: VeriTripConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VeriTripConfig()
    return {
        "name": cfg.paper_title,
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "authors": "Yuting Xu, Jiayi Tian et al. (CAS / Amap / UCAS)",
        "problem": (
            "Retrieval-based travel planning over unstructured multimodal web corpora "
            "with cell-wise VKB verification — not API-centric tool calls."
        ),
        "components": {
            "MRB": "Multimodal Retrieval Base (frozen web snapshot)",
            "VKB": "Verifiable Knowledge Base (evaluator-only ground truth)",
        },
        "metrics": ["DR", "FR", "PRmi", "PRma", "PFR", "AM"],
        "agent_tools": list(AGENT_TOOLS),
        "dataset": table2_dataset_statistics(),
        "evaluation_mapping": table3_evaluation_mapping(),
        "interpretive_boundaries": [
            "Static evidence-sufficient sandbox — not live booking.",
            "Penalizes parametric hallucination via VKB cell checks.",
        ],
        "snapshot_window": f"{cfg.snapshot_start} .. {cfg.snapshot_end}",
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table2": table2_dataset_statistics(),
        "table3": table3_evaluation_mapping(),
        "table4_main": table4_main_results(),
        "table5_visual": table5_visual_ablation(),
        "table6_noisy": table6_noisy_mrb(),
        "constraints": constraint_table(),
    }


def demo_query() -> VeriTripQuery:
    return VeriTripQuery(
        query_id="demo-changsha-001",
        text=(
            "Plan a solo trip from Guangzhou to Changsha 2025-10-28 to 2025-10-30. "
            "Budget 3000 RMB. Visit the museum shown in the attached cropped bronze photo. "
            "Include high-speed rail and local Hunan cuisine."
        ),
        image_path="anchors/changsha_bronze_crop.jpg",
        start_city="Guangzhou",
        target_city="Changsha",
        start_date="2025-10-28",
        end_date="2025-10-30",
        people_number=1,
        budget=3000,
        difficulty="medium",
        persona="Solo traveler",
        preferences={
            "p_attraction": "Hunan Provincial Museum",
            "p_meal": "West Lake Vinegar Fish",
            "p_accommodation": "Double room",
            "p_intercity_traffic": "High-speed rail",
        },
    )


def demo_gold_plan() -> dict[str, Any]:
    """Grounded plan matching demo VKB (high FR)."""
    return {
        "Final Result": {
            "transportationTable": [
                {
                    "transportationID": "G1011",
                    "date": "2025-10-28",
                    "departureStation": "Guangzhou South",
                    "arriveStation": "Changsha South",
                    "begin_time": "08:00",
                    "end_time": "14:30",
                    "price_per_person": 553,
                },
                {
                    "transportationID": "G1012",
                    "date": "2025-10-30",
                    "departureStation": "Changsha South",
                    "arriveStation": "Guangzhou South",
                    "begin_time": "15:00",
                    "end_time": "21:00",
                    "price_per_person": 553,
                },
            ],
            "accommodationTable": [
                {
                    "city": "Changsha",
                    "name": "Changsha W Hotel",
                    "check_in": "2025-10-28",
                    "check_out": "2025-10-30",
                    "price_per_night": 680,
                    "room_type": "Double room",
                    "room_number": 1,
                    "breakfast_included": True,
                }
            ],
            "itineraryTable": [
                {
                    "Date": "2025-10-28",
                    "active_item_number": "1",
                    "active_type": "inter_city_transportation",
                    "start_time": "08:00",
                    "end_time": "14:30",
                    "name": "Guangzhou South to Changsha South",
                    "price_per_person": 553,
                    "notes": "G1011",
                },
                {
                    "Date": "2025-10-28",
                    "active_item_number": "2",
                    "active_type": "attraction",
                    "start_time": "15:00",
                    "end_time": "17:30",
                    "name": "Hunan Provincial Museum",
                    "price_per_person": 50,
                    "notes": "Bronze exhibit [doc-hnm-001]",
                },
                {
                    "Date": "2025-10-28",
                    "active_item_number": "3",
                    "active_type": "meal",
                    "start_time": "18:30",
                    "end_time": "20:00",
                    "name": "Old Hunan Tea House",
                    "price_per_person": 120,
                    "notes": "West Lake Vinegar Fish",
                },
                {
                    "Date": "2025-10-29",
                    "active_item_number": "1",
                    "active_type": "attraction",
                    "start_time": "09:00",
                    "end_time": "12:00",
                    "name": "Orange Isle",
                    "price_per_person": 0,
                    "notes": "",
                },
                {
                    "Date": "2025-10-29",
                    "active_item_number": "2",
                    "active_type": "meal",
                    "start_time": "12:30",
                    "end_time": "13:30",
                    "name": "Fire Palace",
                    "price_per_person": 80,
                    "notes": "Hunan cuisine",
                },
                {
                    "Date": "2025-10-30",
                    "active_item_number": "1",
                    "active_type": "inter_city_transportation",
                    "start_time": "15:00",
                    "end_time": "21:00",
                    "name": "Changsha South to Guangzhou South",
                    "price_per_person": 553,
                    "notes": "G1012",
                },
            ],
        }
    }


def demo_hallucinated_plan() -> dict[str, Any]:
    """Fabricated transport ID — low FR."""
    plan = demo_gold_plan()
    plan["Final Result"]["transportationTable"][0]["transportationID"] = "FAKE999"
    return plan


def evaluation_demo() -> dict[str, Any]:
    """End-to-end: MRB tools → evaluate gold vs hallucinated plans."""
    toolset = build_demo_toolset()
    vkb = build_demo_vkb()
    query = demo_query()

    img_hits = toolset.img_search(query.image_path or "")
    doc_hits = toolset.doc_search("Changsha museum bronze")

    gold_scores = evaluate_plan(demo_gold_plan(), query, vkb)
    bad_scores = evaluate_plan(demo_hallucinated_plan(), query, vkb)

    return {
        "mrb": toolset.mrb.stats(),
        "vkb": vkb.stats(),
        "retrieval": {"imgSearch_top": img_hits[:2], "docSearch_top": doc_hits[:2]},
        "gold_plan": gold_scores.to_dict(),
        "hallucinated_plan": bad_scores.to_dict(),
        "headline_table4": table4_main_results()[0],
    }


def knowledge_card() -> dict[str, Any]:
    """Compact facts for agent harness."""
    return {
        "arxiv": "2605.28683",
        "claim": "Planning requires evidence-grounded retrieval, not parametric memory alone.",
        "metrics": {
            "DR": "Delivery rate — valid JSON",
            "FR": "Factual reliability — VKB cell-wise",
            "PFR": "Preference fulfillment rate",
            "PRmi": "Micro pass rate on hard constraints",
            "AM": "Average geographic margin (10 km units, lower better)",
        },
    }
