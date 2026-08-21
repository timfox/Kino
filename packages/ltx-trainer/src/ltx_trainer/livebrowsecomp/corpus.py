"""LiveBrowseComp question records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LiveBrowseCompItem:
    idx: int
    problem: str
    answer: str
    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "problem": self.problem,
            "answer": self.answer,
            "category": self.category,
        }


def item_from_row(row: dict[str, Any], *, category: str | None = None) -> LiveBrowseCompItem:
    return LiveBrowseCompItem(
        idx=int(row["idx"]),
        problem=str(row["problem"]),
        answer=str(row["answer"]),
        category=category or row.get("category"),
    )
