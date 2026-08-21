"""Access information via call-graph paths to changed functions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FunctionInfo:
    name: str
    category: str  # public | private | special
    file: str = ""
    class_name: str | None = None


@dataclass
class AccessInfo:
    """How to reach a changed function from a public entry (Table I)."""

    target: FunctionInfo
    import_line: str
    entry_function: str | None = None
    signature: str = ""
    docstring: str = ""
    hints: str = ""
    call_path: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "target": self.target.name,
            "category": self.target.category,
            "import_line": self.import_line,
            "entry_function": self.entry_function,
            "signature": self.signature,
            "hints": self.hints,
            "call_path": list(self.call_path),
        }


def classify_function(name: str, *, in_public_api: bool | None = None) -> str:
    """Categorize by public API list / naming conventions (§II-C)."""
    if name.startswith("__") and name.endswith("__"):
        return "special"
    if in_public_api is True:
        return "public"
    if in_public_api is False:
        return "private"
    if name.startswith("_") and not name.startswith("__"):
        return "private"
    return "public"


def extract_access_info(
    func: FunctionInfo,
    *,
    callers: list[str] | None = None,
    public_entries: list[str] | None = None,
    top_k: int = 5,
) -> list[AccessInfo]:
    """Stub of Algorithm 1: top-k shortest public entry paths for private funcs."""
    if func.category == "public":
        cls = func.class_name or "Module"
        return [
            AccessInfo(
                target=func,
                import_line=f"from {func.file.replace('/', '.')}.{cls} import {func.name}",
                signature=f"def {func.name}(...)",
                docstring=f"Public API: {func.name}",
            )
        ]
    if func.category == "special":
        cls = func.class_name or "Object"
        return [
            AccessInfo(
                target=func,
                import_line=f"from {func.file.replace('/', '.')}.{cls} import {cls}",
                signature=f"class {cls}",
                hints=f"{func.name} is a special method; invoke via class instance operations",
            )
        ]
    # private: walk callers to public entries
    entries = public_entries or callers or ["validate"]
    results = []
    for i, entry in enumerate(entries[:top_k]):
        path = [entry, func.name]
        results.append(
            AccessInfo(
                target=func,
                import_line=f"from {func.file.replace('/', '.')}.Schema import {entry}",
                entry_function=entry,
                signature=f"def {entry}(self, data: Mapping) -> dict",
                docstring=f"Public entry that reaches {func.name}",
                call_path=path,
            )
        )
        if i == 0 and not callers:
            break
    return results
