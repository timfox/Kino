"""Lightweight EL TBox (atomic GCIs) for NeurOWL CPU demos."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Subsumption:
    """Atomic concept inclusion A ⊑ B."""

    child: str
    parent: str

    def as_axiom(self) -> str:
        return f"{self.child} ⊑ {self.parent}"


@dataclass
class ELOntology:
    """Directed taxonomy edges child → parent (asserted GCIs)."""

    axioms: list[Subsumption] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    def concepts(self) -> set[str]:
        out: set[str] = set()
        for ax in self.axioms:
            out.add(ax.child)
            out.add(ax.parent)
        out.update(self.labels)
        return out

    def parents(self, concept: str) -> set[str]:
        return {ax.parent for ax in self.axioms if ax.child == concept}

    def children(self, concept: str) -> set[str]:
        return {ax.child for ax in self.axioms if ax.parent == concept}

    def adjacency(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = defaultdict(set)
        for ax in self.axioms:
            adj[ax.child].add(ax.parent)
        return adj


def demo_incomplete_pet_ontology() -> ELOntology:
    """Example 1 O′₁ (missing PetCat ⊑ Cat)."""
    return ELOntology(
        axioms=[
            Subsumption("PersianCat", "PetCat"),
            Subsumption("Cat", "Mammal"),
            Subsumption("Mammal", "Animal"),
        ],
        labels={
            "PersianCat": "Persian cat",
            "PetCat": "pet cat",
            "Cat": "cat",
            "Mammal": "mammal",
            "Animal": "animal",
            "Milk": "milk",
        },
    )


def demo_complete_pet_ontology() -> ELOntology:
    """O′₁ plus the missing PetCat ⊑ Cat bridge."""
    o = demo_incomplete_pet_ontology()
    o.axioms.append(Subsumption("PetCat", "Cat"))
    return o


def demo_food_snippet() -> ELOntology:
    """Tiny FoodOn-like taxonomy for negative / hard-neg demos."""
    return ELOntology(
        axioms=[
            Subsumption("apple", "fruit"),
            Subsumption("fruit", "plant_food"),
            Subsumption("pork", "swine_food_product"),
            Subsumption("swine_food_product", "vertebrate_animal_food_product"),
            Subsumption("beef", "vertebrate_animal_food_product"),
        ],
        labels={
            "apple": "apple",
            "fruit": "fruit",
            "plant_food": "plant food product",
            "pork": "pork",
            "beef": "beef",
            "swine_food_product": "swine food product",
            "vertebrate_animal_food_product": "vertebrate animal food product",
            "Grammatorcynus": "Grammatorcynus",
        },
    )
