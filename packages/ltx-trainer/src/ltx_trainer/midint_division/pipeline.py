"""End-to-end demo: algorithm validation + cost model + GPU mapping."""

from __future__ import annotations

from typing import Any

from ltx_trainer.midint_division.benchmarks import positioning_vs_cgbn, summary_anchors, table_1_cgbn_comparison
from ltx_trainer.midint_division.config import MidintDivisionConfig
from ltx_trainer.midint_division.constants import PAPER_EXAMPLES
from ltx_trainer.midint_division.cost_model import estimate_full_mults, full_mult_bounds
from ltx_trainer.midint_division.division import divide, verify_example
from ltx_trainer.midint_division.gpu_mapping import block_layout_card, operation_catalog, variant_mult_dispatch
from ltx_trainer.midint_division.shifted_inverse import shinv


def run_demo(cfg: MidintDivisionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MidintDivisionConfig()
    examples = cfg.validate_examples or PAPER_EXAMPLES
    validations = [
        verify_example(ex["u"], ex["v"], ex["q"], base=ex.get("B", cfg.base_b))
        for ex in examples
    ]
    u0, v0 = examples[0]["u"], examples[0]["v"]
    q, r, div_meta = divide(u0, v0, base=cfg.base_b, use_newton_shinv=cfg.use_newton_shinv)
    w, sh_meta = shinv(v0, div_meta["h"], cfg.base_b)
    cost = estimate_full_mults(
        h=div_meta["h"],
        k=sh_meta.get("k", div_meta["h"] // 2),
        refine_loops=sh_meta.get("trace", {}).get("loop_count", 3),
    )
    layout = block_layout_card(cfg.bits_exp, cfg.sequentialization_q)
    t1 = table_1_cgbn_comparison()
    row = next(r for r in t1 if r["bits_exp"] == cfg.bits_exp)
    return {
        "config": {
            "word_bits": cfg.word_bits,
            "sequentialization_q": cfg.sequentialization_q,
            "bits_exp": cfg.bits_exp,
            "insts_exp": cfg.insts_exp,
        },
        "paper_examples": validations,
        "all_examples_ok": all(v["ok"] for v in validations),
        "sample_division": {"u": u0, "v": v0, "q": q, "r": r, "meta": div_meta},
        "shinv": {"w": w, "meta": sh_meta},
        "cost_model": cost,
        "cost_bounds": full_mult_bounds(),
        "gpu_layout": layout,
        "cuda_ops": operation_catalog(),
        "variant_mult": variant_mult_dispatch(),
        "table_1_row": row,
        "positioning": positioning_vs_cgbn(),
        "summary": summary_anchors(),
    }
