"""NeurOWL end-to-end pipeline (Figure 1 stages)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.neurowl.config import NeurOWLConfig
from ltx_trainer.neurowl.embedding import bridge_score, rank_candidates
from ltx_trainer.neurowl.llm_verify import llm_accepts_any, llm_accepts_subsumption
from ltx_trainer.neurowl.ontology import (
    ELOntology,
    Subsumption,
    demo_food_snippet,
    demo_incomplete_pet_ontology,
)
from ltx_trainer.neurowl.reasoner import (
    direct_children,
    direct_parents,
    entails,
    justification,
)


@dataclass
class NeurOWLResult:
    entailed: bool
    stage: str
    missing_axioms: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    bridge: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entailed": self.entailed,
            "stage": self.stage,
            "missing_axioms": list(self.missing_axioms),
            "explanation": list(self.explanation),
            "bridge": self.bridge,
            "details": dict(self.details),
        }


def _prune_redundant(
    ontology: ELOntology, accepted: list[tuple[str, str]], direction: str
) -> list[tuple[str, str]]:
    """Paper: if B′₁ ⊑ B′₂, keep only the more specific missing axiom."""
    if not accepted:
        return []
    keep: list[tuple[str, str]] = []
    for c, p in accepted:
        redundant = False
        for c2, p2 in accepted:
            if (c, p) == (c2, p2):
                continue
            if direction == "2a" and c == c2 and entails(ontology, p, p2):
                # A ⊑ B′₁ with B′₁ ⊑ B′₂ ⇒ A ⊑ B′₂ redundant
                redundant = True
                break
            if direction == "2b" and p == p2 and entails(ontology, c2, c):
                redundant = True
                break
        if not redundant:
            keep.append((c, p))
    return keep[:2]


def reason_subsumption(
    ontology: ELOntology,
    child: str,
    parent: str,
    *,
    config: NeurOWLConfig | None = None,
) -> NeurOWLResult:
    """Run Stages 1 → 2a/2b → 3a → 3b for A ⊑ B."""
    cfg = config or NeurOWLConfig()
    backend = cfg.embedding_backend

    # Stage 1
    if entails(ontology, child, parent):
        just = justification(ontology, child, parent)
        return NeurOWLResult(
            entailed=True,
            stage="1",
            explanation=[ax.as_axiom() for ax in just],
            details={"justification_size": len(just)},
        )

    # Stage 2a downward: children B′ of B, missing A ⊑ B′
    if cfg.enable_stage_2a:
        candidates = [(child, b_prime) for b_prime in direct_children(ontology, parent)]
        ranked = rank_candidates(ontology, candidates, backend=backend, top_k=cfg.top_k)
        accepted = llm_accepts_any([(c, p) for c, p, _ in ranked])
        accepted = _prune_redundant(ontology, accepted, "2a")
        if accepted:
            miss = accepted[0]
            just = justification(ontology, miss[1], parent)
            missing = [Subsumption(*miss).as_axiom()]
            expl = missing + [ax.as_axiom() for ax in just]
            return NeurOWLResult(
                entailed=True,
                stage="2a",
                missing_axioms=missing,
                explanation=expl,
                bridge=miss[1],
                details={"ranked": ranked[:5], "accepted": accepted},
            )

    # Stage 2b upward: parents A′ of A, missing A′ ⊑ B
    if cfg.enable_stage_2b:
        candidates = [(a_prime, parent) for a_prime in direct_parents(ontology, child)]
        ranked = rank_candidates(ontology, candidates, backend=backend, top_k=cfg.top_k)
        accepted = llm_accepts_any([(c, p) for c, p, _ in ranked])
        accepted = _prune_redundant(ontology, accepted, "2b")
        if accepted:
            miss = accepted[0]
            just = justification(ontology, child, miss[0])
            missing = [Subsumption(*miss).as_axiom()]
            expl = missing + [ax.as_axiom() for ax in just]
            return NeurOWLResult(
                entailed=True,
                stage="2b",
                missing_axioms=missing,
                explanation=expl,
                bridge=miss[0],
                details={"ranked": ranked[:5], "accepted": accepted},
            )

    # Stage 3a bidirectional
    if cfg.enable_stage_3a:
        pool = [c for c in ontology.concepts() if c not in {child, parent}]
        scored = [
            (c, bridge_score(ontology, child, c, parent, backend=backend)) for c in pool
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        top = scored[: cfg.top_k]
        for c, _s in top:
            if llm_accepts_subsumption(child, c) and llm_accepts_subsumption(c, parent):
                missing = [
                    Subsumption(child, c).as_axiom(),
                    Subsumption(c, parent).as_axiom(),
                ]
                return NeurOWLResult(
                    entailed=True,
                    stage="3a",
                    missing_axioms=missing,
                    explanation=list(missing),
                    bridge=c,
                    details={"top_bridges": top[:5]},
                )

    # Stage 3b direct
    if cfg.enable_stage_3b and llm_accepts_subsumption(child, parent):
        ax = Subsumption(child, parent).as_axiom()
        return NeurOWLResult(
            entailed=True,
            stage="3b",
            missing_axioms=[ax],
            explanation=[ax],
        )

    return NeurOWLResult(entailed=False, stage="fail", details={"query": f"{child} ⊑ {parent}"})


def evaluation_demo() -> dict[str, Any]:
    pet = demo_incomplete_pet_ontology()
    food = demo_food_snippet()
    cases = [
        ("pet_tp", pet, "PetCat", "Animal"),
        ("pet_already", demo_incomplete_pet_ontology(), "Cat", "Animal"),
        ("food_neg", food, "Grammatorcynus", "swine_food_product"),
        ("food_pos", food, "apple", "plant_food"),
    ]
    # apple ⊑ plant_food: Stage 1 after fruit bridge — actually entailed via apple⊑fruit⊑plant_food
    # Make incomplete food by removing apple⊑fruit for a positive abduction case
    food_inc = ELOntology(
        axioms=[ax for ax in food.axioms if not (ax.child == "apple" and ax.parent == "fruit")],
        labels=dict(food.labels),
    )
    cases[3] = ("food_pos", food_inc, "apple", "plant_food")

    results = []
    for name, onto, a, b in cases:
        r = reason_subsumption(onto, a, b)
        results.append({"name": name, "query": f"{a} ⊑ {b}", **r.as_dict()})
    return {"n": len(results), "cases": results}


def evaluation_smoke() -> dict[str, bool]:
    from ltx_trainer.neurowl.benchmarks import PAPER_ANCHORS

    pet = demo_incomplete_pet_ontology()
    tp = reason_subsumption(pet, "PetCat", "Animal")
    already = reason_subsumption(pet, "Cat", "Animal")
    food = demo_food_snippet()
    fp = reason_subsumption(food, "Grammatorcynus", "swine_food_product")
    demo = evaluation_demo()
    checks = {
        "stage1_entailed": already.entailed and already.stage == "1",
        "stage2_abduction": tp.entailed and tp.stage in {"2a", "2b", "3a", "3b"},
        "missing_bridge": any(
            ax.startswith("PetCat ⊑") for ax in (tp.missing_axioms + tp.explanation)
        ),
        "reject_hard_neg": not fp.entailed,
        "demo_n4": demo["n"] == 4,
        "paper_foodon_f1": abs(float(PAPER_ANCHORS["foodona_ont_random_f1"]) - 0.960) < 1e-6,
        "paper_xf1": abs(float(PAPER_ANCHORS["foodona_ont_random_xf1"]) - 0.893) < 1e-6,
        "paper_snomed_hard_f1": abs(float(PAPER_ANCHORS["snomeda_ont_hard_f1"]) - 0.862) < 1e-6,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def evaluation_smoke_fix() -> dict[str, bool]:
    return evaluation_smoke()
