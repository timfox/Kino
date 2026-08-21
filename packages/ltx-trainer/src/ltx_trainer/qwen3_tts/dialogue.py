"""Multi-role dialogue script parsing (ComfyUI DialogueInferenceNode parity)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DialogueLine:
    role: str
    text: str
    index: int


@dataclass
class RoleBank:
    """Named voice clone prompts for dialogue synthesis."""

    roles: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, prompt: Any) -> None:
        key = name.strip()
        if not key:
            raise ValueError("role name must be non-empty")
        self.roles[key] = prompt

    def get(self, name: str) -> Any:
        key = name.strip()
        if key not in self.roles:
            raise KeyError(f"role {name!r} not in bank; have {sorted(self.roles)}")
        return self.roles[key]


_LINE_RE = re.compile(r"^\s*([^:\n]+?)\s*:\s*(.+?)\s*$")


def parse_dialogue_script(script: str) -> list[DialogueLine]:
    """
    Parse ``RoleName: utterance`` lines.

    Blank lines and ``#`` comments are ignored.
    """
    lines: list[DialogueLine] = []
    idx = 0
    for raw in script.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            raise ValueError(f"invalid dialogue line (expected 'Role: text'): {raw!r}")
        role, text = m.group(1).strip(), m.group(2).strip()
        if not role or not text:
            raise ValueError(f"empty role or text in line: {raw!r}")
        lines.append(DialogueLine(role=role, text=text, index=idx))
        idx += 1
    if not lines:
        raise ValueError("dialogue script has no lines")
    return lines


def validate_script_roles(script: str, bank: RoleBank) -> list[str]:
    """Return roles referenced in script but missing from bank."""
    parsed = parse_dialogue_script(script)
    missing = sorted({ln.role for ln in parsed if ln.role not in bank.roles})
    return missing
