"""LogiKEy logical pluralism configuration (arXiv:2605.27246)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LogikeyConfig:
    paper_arxiv: str = "2605.27246"
    paper_title: str = (
        "Many Logics, One Methodology: A Plea for Logical Pluralism in Formalised Reasoning"
    )
    authors: str = (
        "Christoph Benzmüller (Bamberg, FU Berlin); Daniel Kirchner (Bamberg); "
        "Luca Pasetto (Luxembourg)"
    )

    host_environment: str = "Isabelle/HOL (primary); Leo-II/III; Rocq studies noted"
    meta_logic: str = "Classical higher-order logic (HOL) — L0"
    default_object_logic: str = "Higher-order modal logic (HOML) — L1"

    main_claim: str = (
        "Logical pluralism at object-logic level inside a unifying meta-logical framework "
        "(LogiKEy); shallow HOML-in-HOL embeddings enable interdisciplinary reuse without "
        "logical imperialism."
    )

    godel_application: str = (
        "Gödel modal ontological argument + modalised mathematics: positive properties form "
        "a modal ultrafilter; mathematical realism forces cardinality beyond finite/countable."
    )

    layer_ids: tuple[str, ...] = ("L0", "L1", "L2", "L3")

    accessibility_axioms: tuple[str, ...] = ("Rrefl", "Rsymm", "Rtrans")  # S5 fragment

    church_exception: str = (
        "Boolean extensionality (Church Ax7σ) fails globally in HOML; recoverable at a "
        "fixed world or under OneWorld."
    )
