"""Paper benchmark tables (arXiv:2605.28721)."""

from __future__ import annotations

from typing import Any

# Table 3: avg@4 on static vs LiveBrowseComp
TABLE3_MODEL_SCORES: list[dict[str, Any]] = [
    {
        "model": "DeepSeek V4 Pro",
        "params": "1.6T",
        "BrowseComp": 61.4,
        "BrowseComp-ZH": 74.6,
        "HLE": 45.9,
        "LiveBrowseComp": 38.3,
    },
    {
        "model": "Kimi-K2.6",
        "params": "1T",
        "BrowseComp": 62.4,
        "BrowseComp-ZH": 74.8,
        "HLE": 34.7,
        "LiveBrowseComp": 31.7,
    },
    {
        "model": "GLM 5.1",
        "params": "754B",
        "BrowseComp": 68.0,
        "BrowseComp-ZH": 73.5,
        "HLE": 43.6,
        "LiveBrowseComp": 33.9,
    },
    {
        "model": "MiniMax M2.5",
        "params": "230B",
        "BrowseComp": 60.4,
        "BrowseComp-ZH": 66.1,
        "HLE": 27.1,
        "LiveBrowseComp": 28.0,
    },
    {
        "model": "Seed 2.0",
        "params": "—",
        "BrowseComp": 77.3,
        "BrowseComp-ZH": 79.2,
        "HLE": 54.8,
        "LiveBrowseComp": 41.5,
    },
    {
        "model": "GPT 5.4",
        "params": "—",
        "BrowseComp": 72.1,
        "BrowseComp-ZH": 75.3,
        "HLE": 51.9,
        "LiveBrowseComp": 43.2,
    },
]

CATEGORY_DISTRIBUTION: list[dict[str, Any]] = [
    {"category": "Movies", "pct": 26},
    {"category": "Entertainment", "pct": 26},
    {"category": "Science & Tech", "pct": 15},
    {"category": "Sports", "pct": 13},
    {"category": "Others", "pct": 7},
    {"category": "Geography", "pct": 7},
    {"category": "Politics", "pct": 5},
    {"category": "Art & Music", "pct": 1},
]

# Table 5 excerpt (per-domain avg@4 on LiveBrowseComp)
TABLE5_PER_DOMAIN: list[dict[str, Any]] = [
    {"model": "GPT 5.4", "Movies": 48.0, "Entertainment": 43.0, "SciTech": 40.0, "Sports": 46.0, "Society": 39.0, "Avg": 43.2},
    {"model": "GLM 5.1", "Movies": 34.3, "Entertainment": 33.5, "SciTech": 18.5, "Sports": 58.5, "Society": 34.5, "Avg": 33.9},
    {"model": "Kimi K2.5", "Movies": 25.0, "Entertainment": 21.8, "SciTech": 30.6, "Sports": 66.7, "Society": 43.2, "Avg": 30.5},
]

CORRELATION: dict[str, float] = {
    "browsecomp_vs_browsecomp_zh_pearson": 0.79,
    "browsecomp_vs_browsecomp_zh_spearman": 0.87,
    "browsecomp_vs_livebrowsecomp_pearson": 0.53,
    "browsecomp_vs_livebrowsecomp_spearman": 0.74,
}

EXAMPLE_QUESTIONS: list[dict[str, Any]] = [
    {
        "type": "CVE multi-hop",
        "temporal_anchor": "2026 CVE server-interface flaw",
        "answer": "2020",
    },
    {
        "type": "Software product name",
        "temporal_anchor": "WHO technical report ~3y before 2024 eclipse",
        "answer": "WorkTime",
    },
]
