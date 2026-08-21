"""Paper anchors and complexity claims (Sec. 4–6)."""

from __future__ import annotations

PAPER_ANCHORS: dict[str, str] = {
    "zero_ary_non_adaptive": "Thm 4.1: gated zero-ary NRP ↔ non-adaptive query algorithms",
    "monadic_dhn": "Thm 5.5: monadic NRP ↔ Deep Homomorphism Networks (total unary)",
    "frontier_guarded_rowid": "Thm 5.14: frontier guarded NRP → monadic NRP on row-id DBs",
    "focq_relu": "Thm 6.1: simply-gated ReLU-FFN NRP ↔ FOCQ flat queries",
    "tc0_ordered": "Thm 6.3 / Cor 6.6: FOCQ = FO+C = uniform TC0 on ordered Boolean structures",
    "example_43": "Ex 4.3: even triangle count via homomorphism aggregation mod 12",
    "example_56": "Ex 5.6: Q△ triangle product-sum beyond 1-WL GNN",
    "gnn_layer": "Sec 5.1: GNN message-passing as monadic NRP rules",
}

COROLLARIES = (
    "Cor 5.8: DHN classifier ↔ gated monadic NRP flat queries",
    "Cor 5.9: gated monadic NRP strictly subsumes UQAFO",
    "Cor 6.5: ReLU-FFN NRP flat queries ⊂ uniform TC0 (general e-databases)",
)

TABLE_I = [
    {"fragment": "zero-ary NRP", "captures": "non-adaptive query algorithms"},
    {"fragment": "monadic NRP", "captures": "Deep Homomorphism Networks"},
    {"fragment": "frontier guarded NRP", "captures": "DHN on row-id normalized DBs"},
    {"fragment": "ReLU-FFN NRP + FOCQ gate", "captures": "FOCQ / uniform TC0 (ordered)"},
]
