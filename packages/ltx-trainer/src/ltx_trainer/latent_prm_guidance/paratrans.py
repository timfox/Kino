"""ParaTrans benchmark metadata (§3, Appendix C)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.latent_prm_guidance.constants import PARATRANS_DIRECTIONS, PARATRANS_TEST_TASKS, REPAIR_ATTEMPTS


def paratrans_card() -> dict[str, Any]:
    return {
        "name": "ParaTrans",
        "test_tasks": PARATRANS_TEST_TASKS,
        "success_criterion": "compile + run + integrated validation vs reference",
        "directions": [dict(d) for d in PARATRANS_DIRECTIONS],
        "translation_pairs": [
            "CUDA→OpenMP",
            "OpenMP→CUDA",
            "Serial→OpenMP",
            "Serial→CUDA",
        ],
        "repair_loop": f"{REPAIR_ATTEMPTS}-attempt self-repair with execution feedback",
        "fingerprint": "(kernel_name, from_api, to_api)",
        "upstream": "Bitan et al. 2025 UniPar / ParaTrans",
    }


def direction_table() -> list[dict[str, Any]]:
    return [dict(d) for d in PARATRANS_DIRECTIONS]
