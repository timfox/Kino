"""Modular scenario package (.mosc) structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MosPackage:
    """A .mosc package: main.xosc + nested simple/modular scenarios."""

    name: str
    main_references: list[dict[str, Any]] = field(default_factory=list)
    simple_files: list[str] = field(default_factory=list)
    nested_mosc: list[str] = field(default_factory=list)

    def file_count(self) -> int:
        return 1 + len(self.simple_files) + len(self.nested_mosc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "main": "main.xosc",
            "references": self.main_references,
            "simple_files": self.simple_files,
            "nested_mosc": self.nested_mosc,
            "total_files": self.file_count(),
        }


def build_mosc_package(
    name: str,
    references: list[dict[str, Any]],
    simple_files: list[str],
    nested_mosc: list[str] | None = None,
) -> MosPackage:
    return MosPackage(
        name=name,
        main_references=references,
        simple_files=simple_files,
        nested_mosc=nested_mosc or [],
    )
