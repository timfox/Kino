"""CUTA4BPM metamodel elements (Fig. 3) as plain dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BlockKind = Literal["Sequence", "Case", "Loop", "Parallel", "MultipleChoice"]
LoopPosition = Literal["begin", "end"]


@dataclass
class Document:
    name: str


@dataclass
class Role:
    name: str
    organizational_unit: str = ""


@dataclass
class SimpleActivity:
    """CUTA SimpleActivity card (subject–predicate–object sentence)."""

    subject: str
    predicate: str
    obj: str
    role: str = ""
    location: str = ""
    time_limit: str = ""
    input_documents: list[str] = field(default_factory=list)
    output_documents: list[str] = field(default_factory=list)
    seq_no: int = 0

    def sentence(self) -> str:
        return f"{self.subject} {self.predicate} {self.obj}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "SimpleActivity",
            "sentence": self.sentence(),
            "role": self.role,
            "location": self.location,
            "time_limit": self.time_limit,
            "input_documents": list(self.input_documents),
            "output_documents": list(self.output_documents),
            "seq_no": self.seq_no,
        }


@dataclass
class Block:
    kind: BlockKind
    children: list[Any] = field(default_factory=list)  # Block | SimpleActivity
    # Case / MultipleChoice
    conditions: list[str] = field(default_factory=list)
    # Loop
    loop_condition: str = ""
    loop_position: LoopPosition = "begin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "children": [
                c.to_dict() if hasattr(c, "to_dict") else c for c in self.children
            ],
            "conditions": list(self.conditions),
            "loop_condition": self.loop_condition,
            "loop_position": self.loop_position,
        }


@dataclass
class CutaProcess:
    """Top-level CUTA4BPM workflow."""

    name: str
    company_pool: str
    root: Block | SimpleActivity
    roles: list[Role] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        root = self.root.to_dict() if hasattr(self.root, "to_dict") else self.root
        return {
            "name": self.name,
            "company_pool": self.company_pool,
            "root": root,
            "roles": [{"name": r.name, "ou": r.organizational_unit} for r in self.roles],
            "documents": [d.name for d in self.documents],
        }
