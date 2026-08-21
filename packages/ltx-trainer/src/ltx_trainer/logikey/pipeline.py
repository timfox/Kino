"""Framework card and benchmarks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.logikey.config import LogikeyConfig
from ltx_trainer.logikey.layout import LIMITATIONS, LOGIKEY_LAYERS, PIPELINE_STAGES
from ltx_trainer.logikey.mock import evaluation_smoke
from ltx_trainer.logikey.tables import (
    church_postulates_homl,
    embedded_logics_portfolio,
    headline_results,
    logikey_layer_table,
    pluralism_vs_imperialism,
)


def framework_card(cfg: LogikeyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LogikeyConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "problem": (
            "Proof-assistant ecosystems trend toward a single foundational logic "
            "(logical imperialism), hindering interdisciplinary reuse when object-logics "
            "must differ (metaphysics, deontic, free logic)."
        ),
        "proposal": cfg.main_claim,
        "methodology": "LogiKEy",
        "layers": logikey_layer_table(),
        "host": cfg.host_environment,
        "meta_logic": cfg.meta_logic,
        "object_logic_default": cfg.default_object_logic,
        "godel_line": cfg.godel_application,
        "accessibility_axioms": list(cfg.accessibility_axioms),
        "church_exception": cfg.church_exception,
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "layers": logikey_layer_table(),
        "pluralism_contrast": pluralism_vs_imperialism(),
        "church_postulates": church_postulates_homl(),
        "embedded_logics": embedded_logics_portfolio(),
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return evaluation_smoke()
