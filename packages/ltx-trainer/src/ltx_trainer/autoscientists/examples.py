"""Toy demos for AUTOSCIENTISTS workflow and benchmarks."""

from __future__ import annotations

from ltx_trainer.autoscientists.state import SharedState
from ltx_trainer.autoscientists.workflow import compare_to_autoresearch_baseline, run_execution_cycle


def demo_gpt_speedup() -> dict:
    return compare_to_autoresearch_baseline()


def demo_parallel_teams() -> dict:
    state = SharedState(task="gpt_nanochat val_bpb", champion_metric=0.998)
    result = run_execution_cycle(state, max_experiments=60, seed=7, lower_is_better=True)
    return {
        "keeps": result.keeps,
        "experiments": result.experiments,
        "final_bpb": result.state.champion_metric,
        "teams": list(result.state.teams.keys()),
        "ok": result.keeps >= 1 and result.state.champion_metric < 0.998,
    }


def demo_noise_gate_confirm() -> dict:
    from ltx_trainer.autoscientists.noise_gate import NoiseGate

    gate = NoiseGate(sigma=0.001, band_multiplier=2.0)
    clear_win = gate.promote(0.01)
    borderline = gate.promote(0.0015, confirm_fn=lambda: True)
    reject = gate.promote(0.0015, confirm_fn=lambda: False)
    return {"clear_win": clear_win, "borderline_confirmed": borderline, "borderline_rejected": reject}
