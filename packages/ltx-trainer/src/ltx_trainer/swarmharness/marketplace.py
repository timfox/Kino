"""Skill marketplace stub — per-skill credit pricing (Sec. 5.6 future direction)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillListing:
    node_id: str
    skill: str
    credit_per_call: float
    vram_gb: float = 0.0
    trust: float = 0.0


@dataclass
class SkillMarketplace:
    """Nodes publish credit-per-call; submitters compare before dispatch."""

    listings: list[SkillListing] = field(default_factory=list)

    def publish(self, listing: SkillListing) -> None:
        self.listings = [l for l in self.listings if not (l.node_id == listing.node_id and l.skill == listing.skill)]
        self.listings.append(listing)

    def quote(self, skill: str, *, max_price: float | None = None) -> list[SkillListing]:
        candidates = [l for l in self.listings if l.skill == skill]
        if max_price is not None:
            candidates = [l for l in candidates if l.credit_per_call <= max_price]
        return sorted(candidates, key=lambda l: (l.credit_per_call, -l.trust))

    def best_quote(self, skill: str) -> SkillListing | None:
        quotes = self.quote(skill)
        return quotes[0] if quotes else None

    def summary(self) -> dict[str, Any]:
        by_skill: dict[str, int] = {}
        for l in self.listings:
            by_skill[l.skill] = by_skill.get(l.skill, 0) + 1
        return {"listings": len(self.listings), "skills": by_skill}
