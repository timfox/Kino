"""Ensure repo root is on ``sys.path`` for ``gopex_datasets.sphere360`` imports."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root() -> Path:
    root = Path(__file__).resolve().parents[6]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
