"""Table 1 — main 2D face databases (IJCIT survey)."""

from __future__ import annotations

from typing import Any

TABLE1_2D_DATABASES: list[dict[str, Any]] = [
    {
        "name": "Yale",
        "rgb": False,
        "images": 165,
        "persons": 15,
        "size": "320×243",
        "variations": ["i", "e"],
    },
    {
        "name": "AT&T (ORL)",
        "rgb": False,
        "images": 400,
        "persons": 40,
        "size": "92×112",
        "variations": ["i", "a"],
        "images_per_person": 10,
    },
    {
        "name": "XM2VTS",
        "rgb": True,
        "images": 2360,
        "persons": 295,
        "size": "576×720",
        "variations": ["p"],
    },
    {
        "name": "AR",
        "rgb": True,
        "images": 4000,
        "persons": 126,
        "size": "576×768",
        "variations": ["o", "e", "i", "a"],
    },
    {
        "name": "CVL",
        "rgb": True,
        "images": 798,
        "persons": 114,
        "size": "640×480",
        "variations": ["e", "p"],
    },
    {
        "name": "Oulu Physics",
        "rgb": True,
        "images": 2000,
        "persons": 125,
        "size": "428×569",
        "variations": ["i"],
    },
    {
        "name": "JAFFE",
        "rgb": False,
        "images": 213,
        "persons": 10,
        "size": "256×256",
        "variations": ["e"],
    },
    {
        "name": "FEI",
        "rgb": True,
        "images": 2800,
        "persons": 200,
        "size": "640×480",
        "variations": ["p", "e", "i"],
    },
    {
        "name": "LFW",
        "rgb": True,
        "images": 13233,
        "persons": 5749,
        "size": "250×250",
        "variations": ["p", "i", "e", "o", "a"],
    },
]


def table_face_databases() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE1_2D_DATABASES]
