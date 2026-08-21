"""TSC-Bench paper table anchors (Table 1–2)."""

from __future__ import annotations

from typing import Any


def table1_rows() -> dict[str, dict[str, float]]:
    """Table 1 TSC-Bench averages (IoU, BDE, Unipercent, Overall)."""
    return {
        "Gemini2.5-Pro": {
            "iou": 0.187,
            "bde": 0.246,
            "iaa": 0.547,
            "iqa": 0.575,
            "ista": 0.437,
            "aesthetic": 0.815,
            "storytelling": 0.621,
            "overall": 0.621,
        },
        "GPT-5": {
            "iou": 0.168,
            "bde": 0.232,
            "iaa": 0.550,
            "iqa": 0.586,
            "ista": 0.438,
            "aesthetic": 0.810,
            "storytelling": 0.609,
            "overall": 0.609,
        },
        "Qwen3-VL-32B": {
            "iou": 0.312,
            "bde": 0.150,
            "iaa": 0.511,
            "iqa": 0.560,
            "ista": 0.415,
            "aesthetic": 0.809,
            "storytelling": 0.580,
            "overall": 0.580,
        },
        "Qwen3-VL-4B-SFT": {
            "iou": 0.457,
            "bde": 0.106,
            "iaa": 0.543,
            "iqa": 0.593,
            "ista": 0.449,
            "aesthetic": 0.812,
            "storytelling": 0.578,
            "overall": 0.578,
        },
        "ShotCrop3": {
            "iou": 0.544,
            "bde": 0.087,
            "iaa": 0.554,
            "iqa": 0.600,
            "ista": 0.455,
            "aesthetic": 0.826,
            "storytelling": 0.623,
            "overall": 0.623,
        },
    }


def table1_ours() -> dict[str, float]:
    return dict(table1_rows()["ShotCrop3"])


def table1_gpt5() -> dict[str, float]:
    return dict(table1_rows()["GPT-5"])


def table2_ablation() -> list[dict[str, Any]]:
    """Table 2 training stage / reward ablation."""
    return [
        {"model": "Base (SFT)", "s1": False, "s2": False, "iou": False, "ratio": False, "aes": False,
         "iou_val": 0.457, "bde": 0.106, "overall": 0.578},
        {"model": "Base", "s1": False, "s2": False, "iou": False, "ratio": False, "aes": False,
         "iou_val": 0.293, "bde": 0.159, "overall": 0.422},
        {"model": "+ CoT-SFT", "s1": True, "s2": False, "iou": False, "ratio": False, "aes": False,
         "iou_val": 0.498, "bde": 0.096, "overall": 0.695},
        {"model": "+ Semi-SFT", "s1": True, "s2": True, "iou": False, "ratio": False, "aes": False,
         "iou_val": 0.512, "bde": 0.093, "overall": 0.702},
        {"model": "+ R_IoU", "s1": True, "s2": True, "iou": True, "ratio": False, "aes": False,
         "iou_val": 0.545, "bde": 0.088, "overall": 0.718},
        {"model": "+ R_ratio", "s1": True, "s2": True, "iou": True, "ratio": True, "aes": False,
         "iou_val": 0.545, "bde": 0.088, "overall": 0.721},
        {"model": "+ R_aes (full)", "s1": True, "s2": True, "iou": True, "ratio": True, "aes": True,
         "iou_val": 0.545, "bde": 0.087, "overall": 0.725},
    ]


def dataset_catalog() -> dict[str, Any]:
    return {
        "name": "TSC-Bench",
        "train": 6400,
        "test": 1200,
        "domains": ["travel", "street", "cinematic", "professional"],
        "shots": ["medium", "close_up", "establishing"],
    }
