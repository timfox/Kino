"""Spider 2.0–Snow-0212 result anchors (Tables 2–5)."""

from __future__ import annotations

PAPER_ANCHORS: dict[str, str] = {
    "hint_bank": "Persistent H = Hsyn ∪ Hsem; learning outside inference path",
    "strategy_layer": "Trigger → competing strategies with recency + eval_stats",
    "two_stage_inference": "Logic Planning then SQL Synthesis (Sec. 5.4)",
    "development_phase": "113 supervised Spider 2.0–Snow examples bootstrap bank",
    "deployment_phase": "Compiler syntax batch + human-in-the-loop semantic refresh",
    "cross_model_transfer": "Same Hint Bank lifts Doubao/GPT-5 without retraining",
    "attribution_ablation": "Table 4: +10.39 pp pass rate from eval_stats credibility",
}

TABLE_2_MAIN = [
    {
        "backbone": "Doubao-2.0-lite",
        "config": "Vanilla",
        "pass_rate": 0.2942,
        "pass_at_4": 0.4602,
        "syntax_pass": 0.6217,
        "avg_critics": 2.854,
    },
    {
        "backbone": "Doubao-2.0-lite",
        "config": "Tahoe",
        "pass_rate": 0.4912,
        "pass_at_4": 0.6460,
        "syntax_pass": 0.9646,
        "avg_critics": 0.575,
    },
    {
        "backbone": "GPT-5",
        "config": "Vanilla",
        "pass_rate": 0.4270,
        "pass_at_4": 0.6106,
        "syntax_pass": 0.7456,
        "avg_critics": 2.148,
    },
    {
        "backbone": "GPT-5",
        "config": "Tahoe",
        "pass_rate": 0.5885,
        "pass_at_4": 0.7876,
        "syntax_pass": 0.9912,
        "avg_critics": 0.602,
    },
    {
        "backbone": "GPT-5.5",
        "config": "Vanilla",
        "pass_rate": 0.6195,
        "pass_at_4": 0.7257,
        "syntax_pass": 0.9624,
        "avg_critics": 2.790,
    },
    {
        "backbone": "GPT-5.5",
        "config": "Tahoe",
        "pass_rate": 0.7942,
        "pass_at_4": 0.8761,
        "syntax_pass": 1.0000,
        "avg_critics": 0.124,
    },
]

TABLE_3_HELD_OUT = [
    {"config": "Vanilla", "pass_rate": 0.5616, "pass_at_4": 0.6590, "syntax_pass": 0.9021, "avg_critics": 1.184},
    {"config": "SQLGenie-style RAG", "pass_rate": 0.5530, "pass_at_4": 0.6544, "syntax_pass": 0.8577, "avg_critics": 0.446},
    {"config": "Tahoe", "pass_rate": 0.5806, "pass_at_4": 0.6728, "syntax_pass": 0.9914, "avg_critics": 0.203},
]

TABLE_4_ATTRIBUTION = [
    {"config": "Tahoe w/o Attribution", "pass_rate": 0.6903, "pass_at_4": 0.7965, "avg_critics": 0.334},
    {"config": "Tahoe with Attribution", "pass_rate": 0.7942, "pass_at_4": 0.8761, "avg_critics": 0.124},
]

TABLE_5_BY_CATEGORY = [
    {"category": "sf_other", "n": 6, "vanilla_pass_at_4": 0.8333, "tahoe_pass_at_4": 0.8333},
    {"category": "sf_local", "n": 36, "vanilla_pass_at_4": 0.7222, "tahoe_pass_at_4": 0.8333},
    {"category": "sf_bq", "n": 71, "vanilla_pass_at_4": 0.7183, "tahoe_pass_at_4": 0.9014},
]

PARADIGM_COMPARISON = [
    {"paradigm": "Vanilla LLMs", "latency": "Low", "continuous_evolution": "Low", "model_agnostic": "N/A", "accuracy": "Low"},
    {"paradigm": "SFT", "latency": "Low", "continuous_evolution": "Low", "model_agnostic": "Low", "accuracy": "High"},
    {"paradigm": "Test-Time Scaling", "latency": "High", "continuous_evolution": "Low", "model_agnostic": "High", "accuracy": "High"},
    {"paradigm": "Documentation RAG", "latency": "Low", "continuous_evolution": "Medium", "model_agnostic": "High", "accuracy": "Medium"},
    {"paradigm": "Tahoe", "latency": "Low", "continuous_evolution": "High", "model_agnostic": "High", "accuracy": "High"},
]


def benchmarks_bundle() -> dict[str, object]:
    return {
        "anchors": PAPER_ANCHORS,
        "table_2": TABLE_2_MAIN,
        "table_3_held_out": TABLE_3_HELD_OUT,
        "table_4_attribution": TABLE_4_ATTRIBUTION,
        "table_5_categories": TABLE_5_BY_CATEGORY,
        "paradigm_comparison": PARADIGM_COMPARISON,
    }
