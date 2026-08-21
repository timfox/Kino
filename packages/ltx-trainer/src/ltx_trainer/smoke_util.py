"""Helpers to load sibling modules without executing package ``__init__.py`` (avoids torch)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_sibling(caller_file: str | Path, module: str, *, qualname: str | None = None) -> ModuleType:
    """Load ``<pkg>/<module>.py`` next to ``caller_file`` via file location only."""
    root = Path(caller_file).resolve().parent
    path = root / f"{module}.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    pkg = root.name
    name = qualname or f"ltx_trainer.{pkg}.{module}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
