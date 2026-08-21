"""Task taxonomy and benchmark scope (Sec. 3.2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.longav_compass.config import ApplicationScenario, ComplexityLevel, LongAVCompassConfig, X2AVTask


def task_coverage_table() -> list[dict[str, Any]]:
    """Table 2 — task coverage."""
    return [
        {"task": X2AVTask.T2AV.value, "samples": 128, "events": 879, "shots": 2115, "input": "S (script)"},
        {"task": X2AVTask.I2AV.value, "samples": 115, "events": 807, "shots": 1989, "input": "RI+S"},
        {"task": X2AVTask.V2AV.value, "samples": 41, "events": 235, "shots": 731, "input": "RV+S"},
    ]


def scenario_complexity_notes() -> dict[str, str]:
    return {
        ApplicationScenario.VLOG.value: "Informal user-generated minute-long content",
        ApplicationScenario.CONTENT_CREATOR.value: "Structured creator content (comic drama, AI shorts)",
        ApplicationScenario.PERFORMANCE_ADS.value: "E-commerce / conversion campaigns (most discriminative)",
        ApplicationScenario.BRAND_ADS.value: "Large-scale brand marketing narratives",
        ComplexityLevel.L1.value: "Multiple entities or simple short-range interactions",
        ComplexityLevel.L2.value: "Multi-event structures and cross-event transitions",
        ComplexityLevel.L3.value: "Multi-actor interactions and longer-range dependencies",
        ComplexityLevel.L4.value: "Causal chains, physical plausibility, demanding closure",
    }


def benchmark_comparison_table() -> list[dict[str, Any]]:
    """Table 1 — vs prior video / audio-visual benchmarks."""
    return [
        {
            "benchmark": "MSVBench",
            "samples": 276,
            "T2AV": False,
            "I2AV": False,
            "V2AV": False,
            "unified_x2av": False,
            "avg_duration_gt_1min": False,
        },
        {
            "benchmark": "T2AV-Compass",
            "samples": 500,
            "T2AV": True,
            "I2AV": False,
            "V2AV": False,
            "unified_x2av": False,
            "avg_duration_gt_1min": False,
        },
        {
            "benchmark": "VABench",
            "samples": 1299,
            "T2AV": True,
            "I2AV": True,
            "V2AV": False,
            "unified_x2av": False,
            "avg_duration_gt_1min": False,
        },
        {
            "benchmark": "LongAV-Compass",
            "samples": LongAVCompassConfig().n_samples,
            "T2AV": True,
            "I2AV": True,
            "V2AV": True,
            "unified_x2av": True,
            "avg_duration_gt_1min": True,
        },
    ]
