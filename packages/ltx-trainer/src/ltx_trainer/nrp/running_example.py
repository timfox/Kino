"""Paper running examples: Ex. 4.3, 5.6, GNN layer (Secs. 4–5)."""

from __future__ import annotations

from ltx_trainer.nrp.embeddings import EDatabase, EFact
from ltx_trainer.nrp.engine import NRP, default_acceptance, embedded_query, execute, gated_query
from ltx_trainer.nrp.rules import (
    Atom,
    ConjunctionRule,
    DisjunctionRule,
    TransformationRule,
    constant_transform,
    relu_ffn,
    scalar_indicator,
)


def _triangle_graph_db() -> EDatabase:
    """Simple graph: triangle on {a,b,c} plus optional extra edge."""
    db = EDatabase()
    for u, v in [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")]:
        db.add(EFact("E", (u, v), ()))
        db.add(EFact("E", (v, u), ()))
    for v in ["a", "b", "c", "d"]:
        db.add(EFact("V", (v,), (1.0,)))
    return db


def _labeled_triangle_graph_db() -> EDatabase:
    """R-labeled simple graph for Example 5.6."""
    db = EDatabase()
    edges = [("u", "v1"), ("v1", "v2"), ("v2", "u"), ("u", "v3")]
    for a, b in edges:
        db.add(EFact("E", (a, b), ()))
        db.add(EFact("E", (b, a), ()))
    labels = {"u": (2.0,), "v1": (3.0,), "v2": (1.0,), "v3": (4.0,)}
    for node, emb in labels.items():
        db.add(EFact("V", (node,), emb))
    return db


def example_43_even_triangles_program() -> NRP:
    """Example 4.3: graphs with even triangle count (homomorphism count mod 12)."""
    rules: list = [
        ConjunctionRule(
            "Triangle",
            (),
            (
                Atom("E", ("x", "y")),
                Atom("E", ("y", "z")),
                Atom("E", ("x", "z")),
                Atom("True1", ()),
            ),
            head_dim=1,
        ),
        DisjunctionRule("TriangleOrZero", (), ("Triangle", "True0"), head_dim=1),
        TransformationRule(
            "Ans",
            (),
            "TriangleOrZero",
            scalar_indicator(12.0),
            head_dim=1,
        ),
    ]
    return NRP(tuple(rules), output_relation="Ans")


def example_43_seed_db() -> EDatabase:
    db = _triangle_graph_db()
    db.add(EFact("True1", (), (1.0,)))
    db.add(EFact("True0", (), (0.0,)))
    return db


def example_56_q_triangle_program() -> NRP:
    """Example 5.6: Q△ monadic NRP (triangle embedding products summed at u)."""
    rules: list = [
        ConjunctionRule(
            "Zero",
            ("x",),
            (Atom("Adom", ("x",)), Atom("True0", ())),
            head_dim=1,
        ),
        ConjunctionRule(
            "Triangle",
            ("x",),
            (
                Atom("E", ("x", "y")),
                Atom("E", ("y", "z")),
                Atom("E", ("z", "x")),
                Atom("V", ("x",)),
                Atom("V", ("y",)),
                Atom("V", ("z",)),
            ),
            head_dim=1,
        ),
        DisjunctionRule("Q_triangle", ("x",), ("Triangle", "Zero"), head_dim=1),
    ]
    return NRP(tuple(rules), output_relation="Q_triangle")


def example_56_seed_db() -> EDatabase:
    db = _labeled_triangle_graph_db()
    for v in db.active_domain():
        db.add(EFact("Adom", (v,), ()))
    db.add(EFact("True0", (), (0.0,)))
    return db


def gnn_layer_program(input_dim: int = 1, hidden: int = 4) -> NRP:
    """Sec. 5.1 one GNN layer as monadic NRP (simplified concat + ρ)."""
    w1 = [[1.0, 0.5], [0.5, 1.0], [1.0, 0.0], [0.0, 1.0]][:hidden]
    b1 = [0.0] * hidden
    w2 = [[1.0] + [0.0] * (hidden - 1)]
    b2 = [0.0]
    rho = relu_ffn(w1, b1, w2, b2)
    rules: list = [
        ConjunctionRule("AggrIsolated", ("x",), (Atom("Adom", ("x",)), Atom("True0", ())), head_dim=1),
        ConjunctionRule(
            "AggrConnected",
            ("x",),
            (Atom("E", ("x", "y")), Atom("Lambda_i", ("y",))),
            head_dim=1,
        ),
        DisjunctionRule("Aggr", ("x",), ("AggrConnected", "AggrIsolated"), head_dim=1),
        ConjunctionRule(
            "Concat",
            ("x",),
            (Atom("Lambda_i", ("x",)), Atom("Aggr", ("x",))),
            head_dim=2,
        ),
        TransformationRule("Lambda_i1", ("x",), "Concat", rho, head_dim=1),
    ]
    return NRP(tuple(rules), output_relation="Lambda_i1")


def gnn_seed_db() -> EDatabase:
    db = _labeled_triangle_graph_db()
    for v in db.active_domain():
        db.add(EFact("Adom", (v,), ()))
    db.add(EFact("True0", (), (0.0,)))
    db.add(EFact("Lambda_i", ("u",), (1.0,)))
    db.add(EFact("Lambda_i", ("v1",), (2.0,)))
    db.add(EFact("Lambda_i", ("v2",), (0.5,)))
    db.add(EFact("Lambda_i", ("v3",), (1.5,)))
    return db


def run_example_43() -> dict[str, object]:
    db = example_43_seed_db()
    prog = example_43_even_triangles_program()
    ans_db = execute(db, prog)
    triangle_emb = next((f.embedding for f in ans_db.relation_facts("Triangle")), ())
    ans_emb = next((f.embedding for f in ans_db.relation_facts("Ans")), ())
    accepted = gated_query(db, prog, default_acceptance)
    return {
        "triangle_embedding": triangle_emb,
        "ans_embedding": ans_emb,
        "accepted": accepted,
    }


def run_example_56() -> dict[str, object]:
    db = example_56_seed_db()
    prog = example_56_q_triangle_program()
    embs = embedded_query(db, prog)
    return {"Q_triangle_u": embs.get(("u",)), "all_nodes": embs}


def run_gnn_layer() -> dict[str, object]:
    db = gnn_seed_db()
    prog = gnn_layer_program()
    embs = embedded_query(db, prog)
    return {"lambda_i1": embs}
