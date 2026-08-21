"""Appendix A + Example 5.12 running programs."""

from __future__ import annotations

from ltx_trainer.nrp.embeddings import EDatabase, EFact, stack_combine
from ltx_trainer.nrp.engine import NRP, embedded_query, execute, gated_query, default_acceptance
from ltx_trainer.nrp.rules import (
    Atom,
    ConjunctionRule,
    DisjunctionRule,
    TransformationRule,
    relu_threshold,
)


def example_512_supervision_program() -> NRP:
    """Example 5.12: frontier guarded co-supervision query."""
    rules: list = [
        ConjunctionRule(
            "SPD",
            ("x", "y"),
            (Atom("Supervises", ("x", "y")), Atom("PostDoc", ("y",))),
            head_dim=0,
        ),
        ConjunctionRule(
            "JointPhDCount",
            ("x", "y"),
            (
                Atom("SPD", ("x", "y")),
                Atom("Supervises", ("x", "z")),
                Atom("Supervises", ("y", "z")),
                Atom("PhD", ("z",)),
                Atom("True1", ()),
            ),
            head_dim=1,
        ),
        ConjunctionRule(
            "MainPhDCount",
            ("x", "y"),
            (
                Atom("SPD", ("x", "y")),
                Atom("Supervises", ("x", "z")),
                Atom("PhD", ("z",)),
                Atom("True1", ()),
            ),
            head_dim=1,
        ),
        ConjunctionRule(
            "CoSupStats",
            ("x", "y"),
            (Atom("JointPhDCount", ("x", "y")), Atom("MainPhDCount", ("x", "y"))),
            head_dim=2,
            combine_fn=stack_combine,
        ),
        TransformationRule(
            "Good",
            ("x", "y"),
            "CoSupStats",
            relu_threshold(1.0),
            head_dim=1,
        ),
        ConjunctionRule(
            "Ans",
            ("x",),
            (Atom("Good", ("x", "y")),),
            head_dim=1,
        ),
    ]
    return NRP(tuple(rules), output_relation="Ans")


def example_512_seed_db() -> EDatabase:
    db = EDatabase()
    db.add(EFact("Supervises", ("alice", "bob"), ()))
    db.add(EFact("Supervises", ("alice", "carol"), ()))
    db.add(EFact("Supervises", ("bob", "d1"), ()))
    db.add(EFact("Supervises", ("bob", "d2"), ()))
    db.add(EFact("Supervises", ("alice", "d1"), ()))
    db.add(EFact("Supervises", ("alice", "d2"), ()))
    db.add(EFact("PostDoc", ("bob",), ()))
    db.add(EFact("PostDoc", ("carol",), ()))
    db.add(EFact("PhD", ("d1",), ()))
    db.add(EFact("PhD", ("d2",), ()))
    db.add(EFact("True1", (), (1.0,)))
    return db


def appendix_a_graph_program() -> NRP:
    """Appendix A: relational graph construction + neighbor aggregation."""
    rules: list = [
        ConjunctionRule(
            "CNode",
            ("cid",),
            (Atom("Customer", ("cid", "s")), Atom("SegEmb", ("s",))),
            head_dim=1,
        ),
        ConjunctionRule(
            "PNode",
            ("pid",),
            (Atom("Product", ("pid", "k")), Atom("CatEmb", ("k",))),
            head_dim=1,
        ),
        ConjunctionRule(
            "ONode",
            ("oid",),
            (
                Atom("Order", ("oid", "cid", "pid", "t", "r")),
                Atom("CountryEmb", ("t",)),
                Atom("ChanEmb", ("r",)),
            ),
            head_dim=2,
            combine_fn=stack_combine,
        ),
        DisjunctionRule("Node", ("x",), ("CNode", "PNode", "ONode"), head_dim=2),
        ConjunctionRule(
            "Aggr",
            ("u",),
            (Atom("Order", ("u", "cid", "pid", "t", "r")), Atom("Node", ("cid",))),
            head_dim=2,
        ),
    ]
    return NRP(tuple(rules), output_relation="Aggr")


def appendix_a_seed_db() -> EDatabase:
    db = EDatabase()
    db.add(EFact("Customer", ("c1", "retail"), ()))
    db.add(EFact("Customer", ("c2", "biz"), ()))
    db.add(EFact("Product", ("p1", "book"), ()))
    db.add(EFact("Product", ("p2", "tool"), ()))
    db.add(EFact("Order", ("o1", "c1", "p1", "us", "web"), ()))
    db.add(EFact("Order", ("o2", "c2", "p2", "uk", "store"), ()))
    db.add(EFact("SegEmb", ("retail",), (0.2, 0.8)))
    db.add(EFact("SegEmb", ("biz",), (0.9, 0.1)))
    db.add(EFact("CatEmb", ("book",), (1.0, 0.0)))
    db.add(EFact("CatEmb", ("tool",), (0.0, 1.0)))
    db.add(EFact("CountryEmb", ("us",), (1.0, 0.0)))
    db.add(EFact("CountryEmb", ("uk",), (0.0, 1.0)))
    db.add(EFact("ChanEmb", ("web",), (0.7,)))
    db.add(EFact("ChanEmb", ("store",), (0.3,)))
    return db


def run_example_512() -> dict[str, object]:
    db = example_512_seed_db()
    prog = example_512_supervision_program()
    embs = embedded_query(db, prog)
    accepted = gated_query(db, prog, default_acceptance)
    return {"ans": embs, "accepted": accepted}


def run_appendix_a() -> dict[str, object]:
    db = appendix_a_seed_db()
    prog = appendix_a_graph_program()
    result = execute(db, prog)
    nodes = {f.content: f.embedding for f in result.relation_facts("Node")}
    aggr = {f.content: f.embedding for f in result.relation_facts("Aggr")}
    return {"node_count": len(nodes), "aggr_count": len(aggr), "aggr": aggr, "nodes": nodes}
