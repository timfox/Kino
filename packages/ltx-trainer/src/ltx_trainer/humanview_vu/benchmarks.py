"""Table 1 survey scope comparison and Table 6 eval benchmark anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.humanview_vu.config import AWESOME_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL, TABLE1_AXES

# Table 1 — prior surveys vs ours (✓/× from paper)
TABLE1_SURVEY_SCOPES: dict[str, dict[str, bool]] = {
    "[31] Video-LMM post-training": {
        "TG&SG": False,
        "Cap": False,
        "Omni": False,
        "Efficiency": False,
        "Off-Mem": False,
        "Streaming-Mem": False,
        "Text-R": True,
        "O3-R": False,
        "Subfields": False,
        "Train-Data": True,
        "Bench": True,
    },
    "[29] Video understanding w/ LLMs (TCSVT)": {
        "TG&SG": True,
        "Cap": True,
        "Omni": False,
        "Efficiency": False,
        "Off-Mem": False,
        "Streaming-Mem": False,
        "Text-R": True,
        "O3-R": False,
        "Subfields": False,
        "Train-Data": False,
        "Bench": True,
    },
    "[30] VTG with MLLM (TPAMI)": {
        "TG&SG": True,
        "Cap": True,
        "Omni": False,
        "Efficiency": False,
        "Off-Mem": False,
        "Streaming-Mem": False,
        "Text-R": False,
        "O3-R": False,
        "Subfields": False,
        "Train-Data": False,
        "Bench": True,
    },
    "[33] Memory in AI agents": {
        "TG&SG": False,
        "Cap": False,
        "Omni": False,
        "Efficiency": False,
        "Off-Mem": True,
        "Streaming-Mem": True,
        "Text-R": False,
        "O3-R": False,
        "Subfields": False,
        "Train-Data": False,
        "Bench": True,
    },
    "[35] Perception, reason, think, plan": {
        "TG&SG": False,
        "Cap": False,
        "Omni": True,
        "Efficiency": False,
        "Off-Mem": True,
        "Streaming-Mem": False,
        "Text-R": True,
        "O3-R": False,
        "Subfields": False,
        "Train-Data": True,
        "Bench": True,
    },
    "Ours (Human-View W-R-R)": dict.fromkeys(TABLE1_AXES, True),
}

EVAL_BENCHMARK_DIMENSIONS: dict[str, list[str]] = {
    "general_video_understanding": ["Video-MME", "MMBench-Video", "Video-MME v2", "MMWorld", "MVBench"],
    "temporal_spatial": [
        "TempCompass",
        "TOMATO",
        "E.T. Bench",
        "TimeLens",
        "OMTG",
        "MotionBench",
        "STI-Bench",
    ],
    "complex_reasoning": [
        "V-STaR",
        "MINERVA",
        "VideoReasonBench",
        "SEED-Bench-R1",
        "VideoZeroBench",
        "MMR-V",
    ],
    "long_context_streaming": [
        "MLVU",
        "LongVideoBench",
        "LVBench",
        "ALLVB",
        "CG-Bench",
        "StreamBench",
        "OVO-Bench",
        "StreamingVLM",
    ],
    "domain_specific": ["MMVU", "Video-MMMU", "ExpVid", "Video-MMLU", "BEAR"],
    "omnimodal": [
        "WorldSense",
        "OmniVideoBench",
        "LongVALE",
        "LongInsightBench",
        "LVOmniBench",
        "MMOU",
    ],
}


def table1_ours_covers_all_axes() -> bool:
    ours = TABLE1_SURVEY_SCOPES["Ours (Human-View W-R-R)"]
    return all(ours.get(axis, False) for axis in TABLE1_AXES)


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "awesome": AWESOME_URL,
        "table1_axes": list(TABLE1_AXES),
        "table1_survey_scopes": TABLE1_SURVEY_SCOPES,
        "ours_full_coverage": table1_ours_covers_all_axes(),
        "eval_benchmark_dimensions": EVAL_BENCHMARK_DIMENSIONS,
    }
