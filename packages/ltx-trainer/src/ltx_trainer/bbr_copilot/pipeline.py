"""Framework card, demos, smoke (arXiv:2606.03468)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.bbr_copilot.config import BBRCopilotConfig, ProbeBWScenario, MahimahiTestbedConfig
from ltx_trainer.bbr_copilot.metrics import (
    operating_guidelines,
    table_probe_fig8_50ms,
    table_probe_fig9_200ms,
    table_production_stats,
    table_startup_fig7,
)
from ltx_trainer.bbr_copilot.simulation import (
    run_probe_eval,
    run_startup_eval,
    startup_sweep,
)


def framework_card(cfg: BBRCopilotConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BBRCopilotConfig()
    return {
        "name": "BBR-Copilot",
        "paper": cfg.paper_arxiv,
        "title": "When BBR meets live streaming",
        "transport": "QUIC + BBRv1 copilot (padding controller + data generator)",
        "phases": ["Startup", "Drain", "ProbeBW", "ProbeRTT"],
        "trigger": "application-limited AND pacing_gain > 1",
        "benchmarks": ["Verizon-LTE Mahimahi testbed", "5.4 Mbps live GOP"],
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return [
        "Round-based numpy stub — not lsquic/QUIC kernel BBR integration.",
        "BBRv2/v3 loss-based Startup exit not modeled (§5 future work).",
        "Verizon-LTE traces are approximated by fixed Mbps + shallow buffer.",
        "Padding content is filler only — in-flight duplicate recovery not implemented.",
        "Production 570k-flow measurement is a literature anchor, not replayed here.",
    ]


def evaluation_demo(cfg: BBRCopilotConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BBRCopilotConfig()
    startup_cfg = MahimahiTestbedConfig(rtt_ms=200.0)
    probe_50 = ProbeBWScenario(rtt_ms=50.0)
    probe_200 = ProbeBWScenario(rtt_ms=200.0)

    startup_base = run_startup_eval(startup_cfg, use_copilot=False)
    startup_cop = run_startup_eval(startup_cfg, use_copilot=True)
    probe_base_50 = run_probe_eval(probe_50, use_copilot=False)
    probe_cop_50 = run_probe_eval(probe_50, use_copilot=True)

    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "guidelines": operating_guidelines(),
        "startup_200ms": {
            "baseline": startup_base.__dict__,
            "copilot": startup_cop.__dict__,
        },
        "probe_50ms": {
            "baseline_rmse_mbps": probe_base_50.rmse_bps * 8.0 / 1e6,
            "copilot_rmse_mbps": probe_cop_50.rmse_bps * 8.0 / 1e6,
            "window_s": list(probe_50.step_at_s),
        },
        "sweep": startup_sweep(),
        "tables": {
            "startup_fig7": table_startup_fig7(),
            "probe_fig8_50ms": table_probe_fig8_50ms(),
            "probe_fig9_200ms": table_probe_fig9_200ms(),
            "production": table_production_stats(),
        },
    }


def evaluation_smoke(cfg: BBRCopilotConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    startup = demo["startup_200ms"]
    probe = demo["probe_50ms"]
    sweep = demo["sweep"]["startup_sweep"]

    assert startup["copilot"]["exited_startup"] is True
    assert startup["copilot"]["retransmission_ratio"] <= startup["baseline"]["retransmission_ratio"]
    assert probe["copilot_rmse_mbps"] < probe["baseline_rmse_mbps"]

    row_200 = next(r for r in sweep if r["rtt_ms"] == 200.0)
    assert row_200["copilot_exit_ratio"] >= row_200["baseline_exit_ratio"]
    assert row_200["copilot_exit_ratio"] == 1.0

    return {
        "status": "ok",
        "paper": (cfg or BBRCopilotConfig()).paper_arxiv,
        "demo_keys": list(demo.keys()),
    }
