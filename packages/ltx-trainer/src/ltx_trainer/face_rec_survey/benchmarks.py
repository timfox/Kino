"""Table 4 — experimental results cited in IJCIT survey."""

from __future__ import annotations

from typing import Any

TABLE4_EXPERIMENTAL: list[dict[str, Any]] = [
    {"author": "Vankayalapati et al.", "year": 2009, "technique": "CNN", "database": "ORL", "result_pct": 95.0},
    {"author": "Kong, Rui et al.", "year": 2011, "technique": "ICA, SVM", "database": "ORL", "result_pct": 96.0},
    {
        "author": "Bellakhdhar et al.",
        "year": 2013,
        "technique": "Gabor magnitude-phase, PCA, SVM",
        "database": "ORL, FRGCv2",
        "result_pct": 99.9,
    },
    {"author": "Jameel, S.", "year": 2015, "technique": "PCA + DCT in HMM", "database": "ORL", "result_pct": 95.122},
    {
        "author": "Fathima et al.",
        "year": 2015,
        "technique": "Gabor + LDA",
        "database": "AT&T, MIT-India, Faces94",
        "result_pct": 91.01,
        "result_range": "88–94.02%",
    },
    {"author": "Ghorbel et al.", "year": 2016, "technique": "Eigenfaces + DoG", "database": "FERET", "result_pct": 84.26},
    {"author": "Bhaskar, A. et al.", "year": 2016, "technique": "SVM", "database": "Yale", "result_pct": 97.78},
    {
        "author": "Fu et al.",
        "year": 2017,
        "technique": "Guided CNN + metric loss",
        "database": "CASIA-WebFace, LFW",
        "result_pct": 94.5,
        "result_range": "91.9–97.1%",
    },
    {
        "author": "Khan et al.",
        "year": 2018,
        "technique": "PCA",
        "database": "NCR-IIT, video stream",
        "result_pct": 77.5,
        "result_range": "69–86%",
    },
    {
        "author": "Banerjee et al.",
        "year": 2018,
        "technique": "Viola-Jones + generic 3D model",
        "database": "PaSC, CMU Multi-PIE",
        "result_pct": 92.865,
        "result_range": "88.45–97.28%",
    },
]


def table_experimental_results() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE4_EXPERIMENTAL]


def best_result_on(database: str) -> dict[str, Any] | None:
    db = database.lower()
    matches = [r for r in TABLE4_EXPERIMENTAL if db in r["database"].lower()]
    if not matches:
        return None
    return max(matches, key=lambda r: float(r["result_pct"]))
