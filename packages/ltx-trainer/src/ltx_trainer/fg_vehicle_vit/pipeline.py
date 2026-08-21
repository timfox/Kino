"""Framework card, paper tables, training composition, evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fg_vehicle_vit.config import FgVehicleVitConfig
from ltx_trainer.fg_vehicle_vit.inference import pipeline_smoke
from ltx_trainer.fg_vehicle_vit.ltx_plan import ltx_integration_plan
from ltx_trainer.fg_vehicle_vit.stage2 import training_step_demo


def framework_card(cfg: FgVehicleVitConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FgVehicleVitConfig()
    return {
        "name": "FG Vehicle ViT Pipeline",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "stages": [
            f"Stage 1: {cfg.rtdetr_checkpoint} (conf ≥ {cfg.stage1_conf_threshold})",
            f"Stage 2: {cfg.vit_checkpoint} fine-tuned 6-class ViT",
            f"Abstention: softmax < {cfg.abstention_threshold} → unknown",
        ],
        "classes": list(cfg.fine_grained_classes),
        "eval_sites": {
            "in_distribution": "Ann Arbor N. Division bicycle-lane corridor (Oct 2022)",
            "out_of_distribution": "Instrumented bicycle open dataset (Jul–Aug 2023)",
        },
        "fps_decode": 10,
        "open_source": cfg.repo_url,
    }


def table_training_composition() -> list[dict[str, Any]]:
    """Table 1 — training dataset by class and source."""
    return [
        {"class": "Passenger car", "stanford_cars": 9689, "web": 0, "field": 686, "total": 10375, "pct": 62.6},
        {"class": "SUV", "stanford_cars": 2854, "web": 0, "field": 1034, "total": 3888, "pct": 23.4},
        {"class": "Pickup truck", "stanford_cars": 1519, "web": 0, "field": 111, "total": 1630, "pct": 9.8},
        {"class": "Minivan", "stanford_cars": 416, "web": 33, "field": 61, "total": 510, "pct": 3.1},
        {"class": "Large van", "stanford_cars": 0, "web": 68, "field": 24, "total": 92, "pct": 0.6},
        {"class": "Commercial truck", "stanford_cars": 0, "web": 84, "field": 2, "total": 86, "pct": 0.5},
        {"total": 14478, "web_total": 185, "field_total": 1918, "grand_total": 16581, "pct": 100.0},
    ]


def table_in_distribution_metrics() -> list[dict[str, float | int | str]]:
    """Table 2 — in-distribution (n = 3,805)."""
    return [
        {"class": "Passenger car", "n": 1416, "precision": 0.98, "recall": 0.91, "f1": 0.94},
        {"class": "SUV", "n": 1984, "precision": 0.97, "recall": 0.97, "f1": 0.97},
        {"class": "Pickup truck", "n": 217, "precision": 0.96, "recall": 0.92, "f1": 0.94},
        {"class": "Minivan", "n": 126, "precision": 0.91, "recall": 0.91, "f1": 0.91},
        {"class": "Large van", "n": 52, "precision": 0.89, "recall": 0.94, "f1": 0.92},
        {"class": "Commercial truck", "n": 10, "precision": 1.00, "recall": 0.70, "f1": 0.82},
        {"overall_accuracy": 0.94, "n_total": 3805},
    ]


def table_out_of_distribution_metrics() -> list[dict[str, float | int | str]]:
    """Table 3 — OOD (n = 311)."""
    return [
        {"class": "Passenger car", "n": 126, "precision": 0.96, "recall": 0.85, "f1": 0.90},
        {"class": "SUV", "n": 123, "precision": 0.91, "recall": 0.94, "f1": 0.93},
        {"class": "Pickup truck", "n": 27, "precision": 0.96, "recall": 0.96, "f1": 0.96},
        {"class": "Minivan", "n": 16, "precision": 1.00, "recall": 0.56, "f1": 0.72},
        {"class": "Large van", "n": 9, "precision": 1.00, "recall": 0.89, "f1": 0.94},
        {"class": "Commercial truck", "n": 10, "precision": 1.00, "recall": 1.00, "f1": 1.00},
        {"overall_accuracy": 0.89, "n_total": 311},
    ]


def table_abstention_rates() -> list[dict[str, float | str]]:
    """Fig. 4 / Fig. 10 — abstention % by class (ID vs OOD)."""
    return [
        {"class": "Passenger car", "in_distribution": 5.5, "out_of_distribution": 7.1},
        {"class": "SUV", "in_distribution": 1.4, "out_of_distribution": 2.4},
        {"class": "Pickup truck", "in_distribution": 2.4, "out_of_distribution": 0.0},
        {"class": "Minivan", "in_distribution": 2.4, "out_of_distribution": 25.0},
        {"class": "Large van", "in_distribution": 1.9, "out_of_distribution": 11.1},
        {"class": "Commercial truck", "in_distribution": 0.0, "out_of_distribution": 0.0},
    ]


def evaluation_demo(cfg: FgVehicleVitConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FgVehicleVitConfig()
    return {
        "framework": framework_card(cfg),
        "ltx_plan": ltx_integration_plan(cfg),
        "training_step": training_step_demo(cfg),
        "pipeline_smoke": pipeline_smoke(cfg),
        "paper_tables": {
            "training_composition": table_training_composition(),
            "in_distribution": table_in_distribution_metrics(),
            "out_of_distribution": table_out_of_distribution_metrics(),
            "abstention_rates": table_abstention_rates(),
        },
        "delta_accuracy_id_to_ood": cfg.eval_id_accuracy - cfg.eval_ood_accuracy,
    }
