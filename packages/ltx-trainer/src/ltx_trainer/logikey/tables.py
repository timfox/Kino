"""LogiKEy comparison and methodology tables."""

from __future__ import annotations

from typing import Any


def logikey_layer_table() -> list[dict[str, str]]:
    from ltx_trainer.logikey.layout import LOGIKEY_LAYERS

    return [
        {"layer": lid, "role": role, "content": content}
        for lid, role, content in LOGIKEY_LAYERS
    ]


def pluralism_vs_imperialism() -> list[dict[str, str]]:
    return [
        {
            "stance": "Logical pluralism (LogiKEy)",
            "feature": "Object-logic negotiable in design space",
            "consequence": "Explicit L0–L3 layering; compare/exchange logics",
        },
        {
            "stance": "Logical imperialism",
            "feature": "Single foundational logic for all libraries",
            "consequence": "Monoculture, invisible axioms (LEM, choice, 1/0=0, …)",
        },
        {
            "stance": "Principled monism (Zalta PLM)",
            "feature": "Rich unified foundation (AOT, hyperintensionality)",
            "consequence": "Adequate for metaphysics+math but heavy commitments",
        },
        {
            "stance": "Isabelle (original design)",
            "feature": "Minimal Pure meta-logic; multiple object logics",
            "consequence": "Architecture pluralistic; practice often HOL-only",
        },
    ]


def church_postulates_homl() -> list[dict[str, Any]]:
    return [
        {"id": "Ax1–Ax6", "status": "derivable_in_homl_smoke", "note": "Lifted Church postulates"},
        {"id": "Ax7σ", "status": "blocked", "note": "Boolean extensionality — modal anti-substitution"},
        {"id": "Ax7@fixed_world", "status": "recoverable", "note": "Restrict to actual world"},
        {"id": "Ax7@OneWorld", "status": "recoverable", "note": "Single-world collapse to HOL"},
    ]


def embedded_logics_portfolio() -> list[str]:
    return [
        "Higher-order (multi-)modal logic (HOML)",
        "Intuitionistic and multivalued logics",
        "Deontic, conditional, access-control logics",
        "Free logic",
        "Public announcement logic",
        "PLM/AOT (embedded as object-logic in prior work)",
    ]


def headline_results() -> dict[str, str]:
    return {
        "methodology": "LogiKEy: shallow (and emerging deep) object-logic embeddings in HOL",
        "metaphysics": "Gödel modal ontology + modal maths in one framework",
        "cardinality": "Existing mathematical entities force uncountably many positive properties",
        "contrast": "Logical pluralism vs imperialism; complement to PLM monism",
    }
