"""Tables 1–3 excerpts (arXiv:2605.30260)."""

from __future__ import annotations

from typing import Any


def table1_law_fit() -> list[dict[str, Any]]:
    """Goodness-of-fit for Eq. 6 (subset)."""
    return [
        {"model": "Llama3.1-8B-IT", "setting": "Comb.", "r2": 0.987, "mape_pct": 7.057},
        {"model": "Llama3.1-8B-IT", "setting": "Phonebook", "r2": 0.981, "mape_pct": 1.606},
        {"model": "Qwen3-8B-IT", "setting": "Comb.", "r2": 0.983, "mape_pct": 8.320},
        {"model": "Qwen3-8B-IT", "setting": "Phonebook", "r2": 0.990, "mape_pct": 0.476},
    ]


def table2_memft_longcontext_acctok() -> dict[str, list[float]]:
    """Table 2 — Long-Context token accuracy (%) excerpts."""
    return {
        "Llama3.1-8B-IT": {
            "SFT": [27.4, 28.5, 43.6, 45.9, 54.9, 69.5, 78.2, 86.3, 94.7],
            "MemFT-OT": [27.3, 36.4, 45.6, 54.7, 63.6, 70.5, 85.4, 94.7, 100.0],
            "MemFT-SW": [32.5, 37.5, 46.0, 52.3, 56.0, 63.4, 69.1, 76.6, 81.1],
        },
        "Qwen3-8B-IT": {
            "SFT": [17.9, 24.2, 27.8, 31.7, 33.1, 39.8, 40.2, 40.0, 47.7],
            "MemFT-OT": [19.2, 23.6, 29.8, 38.5, 47.5, 56.1, 91.1, 100.0, 100.0],
            "MemFT-SW": [24.7, 29.3, 32.0, 39.4, 52.5, 74.6, 93.5, 94.4, 94.4],
        },
    }


def table2_phonebook_accem() -> dict[str, list[float]]:
    return {
        "Llama3.1-8B-IT": {
            "SFT": [0.50, 3.85, 18.7, 28.0, 37.8, 47.0, 59.3],
            "MemFT-OT": [1.00, 11.2, 31.4, 53.9, 61.0, 73.9, 87.0],
            "MemFT-SW": [1.84, 15.0, 34.0, 45.7, 70.7, 96.1, 100.0],
        },
        "Qwen3-8B-IT": {
            "SFT": [2.32, 17.4, 37.5, 55.5, 84.8, 99.5, 100.0],
            "MemFT-OT": [5.78, 19.1, 36.2, 57.4, 86.1, 98.6, 100.0],
            "MemFT-SW": [8.45, 19.7, 37.8, 58.8, 86.5, 99.5, 100.0],
        },
    }


def table4_exact_memory_scenarios() -> list[dict[str, Any]]:
    """Representative exact-memory domains (Table 4)."""
    return [
        {
            "domain": "Personal credentials",
            "query": "What is the login email and password for the internal portal?",
            "target_note": "Exact email/password string",
        },
        {
            "domain": "Legal compliance",
            "query": "Recite Article 5(1)(a) GDPR wording",
            "target_note": "Verbatim statutory text",
        },
        {
            "domain": "Medical coding",
            "query": "ICD-10 for uncomplicated Type 2 diabetes",
            "target": "E11.9",
        },
        {
            "domain": "Model watermark",
            "query": "Embedded ownership watermark",
            "target": "MEM-2026-LoRA-EXACT-0x7F9A3B-COPYRIGHT",
        },
        {
            "domain": "Cloud configuration",
            "query": "Production AWS S3 log bucket endpoint",
            "target_note": "Full s3:// URI",
        },
        {
            "domain": "Academic citation",
            "query": "LaTeX cross-entropy loss",
            "target_note": "Full \\mathcal{L}_{CE} expression",
        },
        {
            "domain": "Security secret",
            "query": "Memory leakage detection nonce",
            "target": "SECRET-TEST-XXXX-XXXX-XXXX-NONCE-1234",
        },
        {
            "domain": "Software license",
            "query": "Enterprise activation key",
            "target": "ENT-2026-ABCD-EFGH-IJKL-MNOP-QRST",
        },
    ]


def table3_linear_rule_generalization() -> list[dict[str, Any]]:
    """Linear rule f(x,y)=3x+5y+7 on Qwen3-8B."""
    return [
        {"rank": 1, "sft_memory": 83.0, "memft_memory": 95.0, "sft_gen": 19.0, "memft_gen": 34.0},
        {"rank": 2, "sft_memory": 100.0, "memft_memory": 97.0, "sft_gen": 38.0, "memft_gen": 47.0},
        {"rank": 4, "sft_memory": 99.0, "memft_memory": 100.0, "sft_gen": 46.0, "memft_gen": 53.0},
        {"rank": 8, "sft_memory": 100.0, "memft_memory": 99.0, "sft_gen": 39.0, "memft_gen": 49.0},
        {"rank": 16, "sft_memory": 100.0, "memft_memory": 100.0, "sft_gen": 41.0, "memft_gen": 54.0},
    ]
