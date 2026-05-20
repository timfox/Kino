#!/usr/bin/env python3
"""Train the LTX text stack (bridge + aggregates + connectors) with frozen DiT — Phase 1a entrypoint.

Thin wrapper around ``train.py`` using ``configs/ltx2_text_stack_gemma4_bridge.yaml``.
Edit paths in that YAML or pass a custom config::

    python scripts/train_text_stack.py configs/ltx2_text_stack_gemma4_bridge.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
try:
    import _kino_bootstrap  # noqa: F401
except ImportError:
    pass

if __name__ == "__main__":
    from train import main as train_main

    _DEFAULT = _SCRIPTS.parent / "configs" / "ltx2_text_stack_gemma4_bridge.yaml"
    argv = sys.argv[1:]
    if not argv or (argv[0].startswith("-") and not any(a.endswith(".yaml") or a.endswith(".yml") for a in argv)):
        if not _DEFAULT.is_file():
            print(f"Default config missing: {_DEFAULT}", file=sys.stderr)
            raise SystemExit(1)
        sys.argv = [sys.argv[0], str(_DEFAULT), *argv]
    train_main()
