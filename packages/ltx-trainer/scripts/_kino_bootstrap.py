"""Put kino package ``src`` trees on ``sys.path`` (safe for any cwd / relative PYTHONPATH)."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_kino_pythonpath() -> None:
    scripts_dir = Path(__file__).resolve().parent
    trainer_pkg = scripts_dir.parent
    packages = trainer_pkg.parent
    for root in (
        trainer_pkg / "src",
        packages / "ltx-core" / "src",
        packages / "ltx-pipelines" / "src",
    ):
        entry = str(root)
        if root.is_dir() and entry not in sys.path:
            sys.path.insert(0, entry)


bootstrap_kino_pythonpath()

try:
    from ltx_trainer.nvml_safe_cuda import apply_nvml_safe_cuda_patches

    apply_nvml_safe_cuda_patches()
except Exception:
    pass
