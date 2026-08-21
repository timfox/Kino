"""LogiKEy methodology layout constants."""

from __future__ import annotations

LOGIKEY_LAYERS: tuple[tuple[str, str, str], ...] = (
    ("L0", "Meta-logic", "Classical HOL (Church simple type theory)"),
    ("L1", "Object-logic(s)", "HOML, deontic, conditional, free logics (shallow/deep embed)"),
    ("L2", "Domain theories", "Gödel modal ontology + modalised mathematics"),
    ("L3", "Applications", "Computational metaphysics, normative/legal reasoning"),
)

PIPELINE_STAGES: tuple[str, ...] = (
    "select_object_logic",
    "shallow_or_deep_embed_in_hol",
    "formalise_domain_theory_l2",
    "run_application_l3",
    "meta_level_consistency_and_model_find",
    "compare_or_exchange_object_logic",
)

LIMITATIONS: tuple[str, ...] = (
    "Reference stub — not a full Isabelle/HOL or Leo-III reproduction.",
    "Shallow embedding adequacy/completeness only sketched; deep HOML faithfulness TBD.",
    "Gödel cardinality results are finite-toy smokes, not full mechanised Cantor in HOML.",
    "Relative consistency of hybrid HOL+object-logic axioms not formally proved here.",
)
