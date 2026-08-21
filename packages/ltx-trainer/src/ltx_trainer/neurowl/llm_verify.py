"""LLM verification stand-in (prompt learning → YES/NO)."""

from __future__ import annotations

# Plausible atomic GCIs for demo ontologies (paper Example 2 + FoodOn-like).
PLAUSIBLE_SUBSUMPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("PetCat", "Cat"),
        ("PetCat", "Mammal"),
        ("PersianCat", "Cat"),
        ("PersianCat", "Animal"),
        ("PetCat", "Animal"),
        ("Cat", "Animal"),
        ("apple", "fruit"),
        ("apple", "plant_food"),
        ("pork", "swine_food_product"),
        ("pork", "vertebrate_animal_food_product"),
        # Known false (FP case study style)
        # Grammatorcynus ⊑ swine_food_product is NOT here
    }
)


def llm_accepts_subsumption(child: str, parent: str) -> bool:
    """Stage 2/3 LLM check — YES iff in plausible set or obvious lexical is-a."""
    if (child, parent) in PLAUSIBLE_SUBSUMPTIONS:
        return True
    # Reject fish-genus → swine food product style false bridges
    if child == "Grammatorcynus" and "swine" in parent.lower():
        return False
    if child == "Grammatorcynus" and "vertebrate_animal_food" in parent:
        return False
    # Conservative: unknown pairs are rejected (Stage 3b False)
    return False


def llm_accepts_any(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(c, p) for c, p in pairs if llm_accepts_subsumption(c, p)]
