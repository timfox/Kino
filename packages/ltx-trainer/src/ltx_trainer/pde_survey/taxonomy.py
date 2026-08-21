"""Table 1 scenario taxonomy (Sec. 4.2)."""

from __future__ import annotations

from typing import Any


def table1_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "scenario": "NLP Benchmark Contamination",
            "user_types": ["LLM developer"],
            "description": "Public NLP benchmarks in pretraining inflate evaluation via memorization.",
            "security_risks": "Unreliable generalization metrics; test-train overlap.",
            "example_refs": ["11", "62", "63", "6", "45", "2", "41"],
        },
        {
            "scenario": "Personal Data Exposure from Web Crawling",
            "user_types": ["API user"],
            "description": "Web corpora ingest PII/social posts; models may regurgitate on prompt.",
            "security_risks": "Privacy leakage of sensitive personal details.",
            "example_refs": ["32", "66", "56"],
        },
        {
            "scenario": "Copyrighted Content & IP Risks",
            "user_types": ["API provider"],
            "description": "Copyrighted web text in pretraining; verbatim regurgitation risk.",
            "security_risks": "Legal/ethical IP violations in outputs.",
            "example_refs": ["48", "36", "49", "65", "24"],
        },
        {
            "scenario": "Code & Software Security Risks",
            "user_types": ["API provider", "API user"],
            "description": "GitHub/StackOverflow code in pretraining; license and secret leakage.",
            "security_risks": "License violations, insecure or proprietary code snippets.",
            "example_refs": ["64", "54", "33", "62", "40"],
        },
    ]
