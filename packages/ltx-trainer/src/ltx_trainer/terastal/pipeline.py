"""End-to-end Terastal demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.terastal.accelerators import vgg11_layer_latencies_us
from ltx_trainer.terastal.benchmarks import miss_rate_comparison_anchors, summary_anchors, table_i_hardware
from ltx_trainer.terastal.budget import model_virtual_budgets, virtual_deadline
from ltx_trainer.terastal.config import TerastalConfig
from ltx_trainer.terastal.scheduler import ReadyLayer, SchedulerState, schedule_round, terastal_vs_baseline_reduction
from ltx_trainer.terastal.variants import (
    minimum_gamma_for_latency,
    valid_variant_combinations,
    variant_transform_card,
)


def run_demo(cfg: TerastalConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TerastalConfig()
    profile = vgg11_layer_latencies_us()
    layer_lat = [[row["ws_us"], row["os_us"]] for row in profile]
    budgets = model_virtual_budgets(layer_lat, cfg.deadline_s)

    # Build one scheduling round with VGG11 layer 8 (high OS penalty)
    late = profile[7]
    gamma = minimum_gamma_for_latency(late["ws_us"], late["os_us"]) or 2
    v_deadlines = virtual_deadline(0.0, budgets.get("budgets_s", [cfg.deadline_s / len(profile)] * len(profile)))

    ready = [
        ReadyLayer(
            request_id="J1_VGG11",
            model="VGG11",
            layer=8,
            arrival_s=0.0,
            virtual_deadline_s=v_deadlines[7],
            latencies_us={"ws0": late["ws_us"], "os1": late["os_us"], "os2": late["os_us"]},
            variant_latencies_us={
                "ws0": late["ws_us"],
                "os1": late["os_us"] / (gamma * gamma),
                "os2": late["os_us"] / (gamma * gamma),
            },
        )
    ]
    state = SchedulerState(
        time_s=0.0,
        accel_free_at={"ws0": 0.0, "os1": 0.0, "os2": 0.0},
    )
    assignments, state_after = schedule_round(ready, state)

    # Stub miss rates for baseline comparison (Fig. 5 normalized)
    baseline_fcfs = 0.42
    baseline_edf = 0.35
    baseline_dream = 0.38
    terastal_miss = baseline_fcfs * (1.0 - 0.4058)

    return {
        "variant_transform": variant_transform_card(gamma),
        "virtual_budgets": budgets,
        "variant_gamma_layer8": gamma,
        "valid_combinations": valid_variant_combinations(4, cfg.accuracy_threshold)[:5],
        "schedule_round": {
            "assignments": assignments,
            "completed": state_after.completed,
        },
        "miss_rate_stub": {
            "terastal": round(terastal_miss, 4),
            "fcfs": baseline_fcfs,
            "edf": baseline_edf,
            "dream": baseline_dream,
            "reduction_vs_fcfs": round(terastal_vs_baseline_reduction(baseline_fcfs, terastal_miss), 4),
        },
        "hardware": table_i_hardware(),
        "anchors": miss_rate_comparison_anchors(),
        "summary": summary_anchors(),
    }
