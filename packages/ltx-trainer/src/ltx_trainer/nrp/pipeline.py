"""NRP evaluation smoke and demo pipeline."""

from __future__ import annotations

from ltx_trainer.nrp.benchmarks import COROLLARIES, PAPER_ANCHORS, TABLE_I
from ltx_trainer.nrp.config import NrpConfig, PAPER_ARXIV, PAPER_TITLE
from ltx_trainer.nrp.dhn import demo_dhn_triangle
from ltx_trainer.nrp.embeddings import product_combine, stack_combine, sum_aggregate
from ltx_trainer.nrp.extended_examples import run_appendix_a, run_example_512
from ltx_trainer.nrp.focq import WeightedStructure, eval_term, mul, q, rel, sum_term
from ltx_trainer.nrp.integration import framework_card, integration_bundle
from ltx_trainer.nrp.rowid import ordered_row_id_expansion
from ltx_trainer.nrp.running_example import (
    example_43_seed_db,
    run_example_43,
    run_example_56,
    run_gnn_layer,
)


def evaluation_demo() -> dict[str, object]:
    ex43 = run_example_43()
    ex56 = run_example_56()
    gnn = run_gnn_layer()
    ex512 = run_example_512()
    appendix = run_appendix_a()
    dhn = demo_dhn_triangle()
    struct = WeightedStructure(
        domain=("a", "b"),
        relations={"R": {("a",): (1.0,), ("b",): (0.0,)}},
    )
    term = sum_term(("x",), mul(q(2.0), rel("R", ("x",))))
    focq_val = eval_term(term, struct, {})
    rowid = ordered_row_id_expansion(example_43_seed_db())
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "framework": framework_card(),
        "integration": integration_bundle(),
        "anchors": PAPER_ANCHORS,
        "corollaries": COROLLARIES,
        "table_i": TABLE_I,
        "example_43": ex43,
        "example_56": ex56,
        "example_512": ex512,
        "appendix_a": appendix,
        "gnn_layer": gnn,
        "dhn_triangle": dhn,
        "rowid_facts": len(rowid),
        "focq_demo": focq_val,
        "combine_demo": product_combine((2.0, 3.0), (4.0,)),
        "stack_demo": stack_combine((2.0,), (3.0,)),
        "aggregate_demo": sum_aggregate([(1.0,), (2.0,)]),
    }


def evaluation_smoke() -> dict[str, bool]:
    ex43 = run_example_43()
    ex56 = run_example_56()
    gnn = run_gnn_layer()
    ex512 = run_example_512()
    appendix = run_appendix_a()
    dhn = demo_dhn_triangle()
    triangle_count = ex43["triangle_embedding"][0] if ex43["triangle_embedding"] else 0.0
    checks = {
        "example_43_triangle_positive": triangle_count > 0,
        "example_43_single_triangle_not_accepted": () not in ex43["accepted"],
        "example_56_u_positive": (ex56["Q_triangle_u"] or (0.0,))[0] > 0,
        "example_512_alice_accepted": ("alice",) in ex512["accepted"],
        "appendix_a_nodes": appendix["node_count"] >= 4,
        "appendix_a_aggr": appendix["aggr_count"] >= 1,
        "dhn_u_positive": (dhn.get("u_embedding") or (0.0,))[0] > 0,
        "gnn_layer_produces_all_nodes": len(gnn["lambda_i1"]) >= 4,
        "product_combine_ok": product_combine((2.0, 3.0), (4.0,)) == (8.0, 3.0),
        "stack_combine_ok": stack_combine((2.0,), (3.0,)) == (2.0, 3.0),
        "sum_aggregate_ok": sum_aggregate([(1.0, 2.0), (3.0, 4.0)]) == (4.0, 6.0),
        "framework_card_ok": framework_card()["paper"]["arxiv"] == PAPER_ARXIV,
    }
    checks["all_pass"] = all(checks.values())
    return checks
