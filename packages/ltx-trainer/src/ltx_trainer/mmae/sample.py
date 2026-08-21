"""MMAE sample and rubric schema (Appendix B JSON format)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltx_trainer.mmae.config import RubricCategory


@dataclass
class Rubric:
    category: RubricCategory
    question: str
    right_choice: str
    wrong_choices: list[str]

    def all_choices(self) -> list[str]:
        return [self.right_choice, *self.wrong_choices]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "question": self.question,
            "right_choice": self.right_choice,
            "wrong_choices": list(self.wrong_choices),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rubric:
        cat = RubricCategory(data["category"])
        return cls(
            category=cat,
            question=str(data["question"]),
            right_choice=str(data["right_choice"]),
            wrong_choices=[str(x) for x in data["wrong_choices"]],
        )


@dataclass
class MMAESample:
    sample_id: str
    complexity: str
    modality: str
    granularity: list[str]
    operations: list[dict[str, str]]
    instruction: str
    tags: list[list[str]] = field(default_factory=list)
    rubrics: list[Rubric] = field(default_factory=list)
    audio_paths: list[str] = field(default_factory=list)
    rounds: list[str] | None = None
    duration_sec: float | None = None

    @property
    def num_rubrics(self) -> int:
        return len(self.rubrics)

    @property
    def num_if_rubrics(self) -> int:
        return sum(1 for r in self.rubrics if r.category == RubricCategory.INSTRUCTION_FOLLOWING)

    @property
    def num_cr_rubrics(self) -> int:
        return sum(1 for r in self.rubrics if r.category == RubricCategory.CONSISTENCY)

    def to_dict(self) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction},
                    *(
                        {"type": "audio", "audio_url": p}
                        for p in self.audio_paths
                    ),
                ],
            }
        ]
        out: dict[str, Any] = {
            "id": self.sample_id,
            "complexity": self.complexity,
            "modality": self.modality,
            "granularity": list(self.granularity),
            "operations": list(self.operations),
            "messages": messages,
            "tags": self.tags,
            "rubrics": [r.to_dict() for r in self.rubrics],
        }
        if self.rounds:
            out["rounds"] = list(self.rounds)
        if self.duration_sec is not None:
            out["duration_sec"] = self.duration_sec
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MMAESample:
        instruction = ""
        audio_paths: list[str] = []
        for msg in data.get("messages", []):
            if msg.get("role") != "user":
                continue
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    instruction = str(part.get("text", ""))
                elif part.get("type") == "audio":
                    audio_paths.append(str(part.get("audio_url", "")))
        rubrics = [Rubric.from_dict(r) for r in data.get("rubrics", [])]
        if not instruction and data.get("instruction"):
            instruction = str(data["instruction"])
        if not audio_paths and data.get("audio_paths"):
            audio_paths = [str(p) for p in data["audio_paths"]]
        return cls(
            sample_id=str(data.get("id") or data.get("sample_id", "")),
            complexity=str(data["complexity"]),
            modality=str(data["modality"]),
            granularity=[str(g) for g in data.get("granularity", [])],
            operations=list(data.get("operations", [])),
            instruction=instruction,
            tags=list(data.get("tags", [])),
            rubrics=rubrics,
            audio_paths=audio_paths,
            rounds=list(data["rounds"]) if "rounds" in data else None,
            duration_sec=float(data["duration_sec"]) if "duration_sec" in data else None,
        )


def load_sample_json(path: str | Path) -> MMAESample:
    raw = Path(path).read_text(encoding="utf-8")
    return MMAESample.from_dict(json.loads(raw))
