"""End-to-end demo: workflows + tables + interpretive framework."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deopt_reopt.benchmarks import conv2d_highlight, summary_anchors, table_2_comparison
from ltx_trainer.deopt_reopt.config import DeoptReoptConfig
from ltx_trainer.deopt_reopt.kernels import divergent_kernels, interpretive_group_note
from ltx_trainer.deopt_reopt.metrics import normalize_loc
from ltx_trainer.deopt_reopt.statistics import iterative_significance_summary, single_shot_significance_summary
from ltx_trainer.deopt_reopt.workflows import deoptimization_contract, single_shot_phases, workflow_card


_SAMPLE_CPU = """
// NEON-blocked conv2d microkernel (excerpt)
#pragma omp parallel for
for (int n = 0; n < N; ++n) {
  float32x4_t acc = vdupq_n_f32(0.f);
  for (int c = 0; c < C; ++c)
    acc = vfmaq_f32(acc, load_filter_transpose(...), load_input(...));
  store_output(n, acc);
}
"""


def run_demo(cfg: DeoptReoptConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DeoptReoptConfig()
    ss = single_shot_significance_summary()
    it = iterative_significance_summary()
    conv = conv2d_highlight()
    return {
        "config": {
            "model": cfg.model,
            "workflow": cfg.workflow,
            "trials": cfg.trials,
            "kernels": cfg.kernels,
        },
        "workflows": workflow_card(),
        "deoptimization_contract": deoptimization_contract(),
        "single_shot_phases": {
            "direct": single_shot_phases("direct"),
            "deopt_reopt": single_shot_phases("deopt_reopt"),
            "direct_3": single_shot_phases("direct_3"),
        },
        "divergent_kernels": divergent_kernels(),
        "conv2d_highlight": conv,
        "single_shot_stats": ss,
        "iterative_stats": it,
        "table_2_single_shot": table_2_comparison("single_shot"),
        "table_2_iterative": table_2_comparison("iterative"),
        "sample_loc": {
            "input_cpp_normalized": normalize_loc(_SAMPLE_CPU),
            "note": "LOC proxy for deoptimization selection in Iterative",
        },
        "group_notes": {k: interpretive_group_note(k) for k in divergent_kernels()},
        "summary": summary_anchors(),
        "all_bh_claims_consistent": (
            len(ss["deopt_reopt_wins_bh"]) == 5 and len(ss["direct_wins_bh"]) == 3
        ),
    }
