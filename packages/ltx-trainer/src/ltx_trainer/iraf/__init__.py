"""IRAF — interference-resilient adaptive fusion (arXiv:2606.06559)."""

from ltx_trainer.iraf.config import IrafConfig
from ltx_trainer.iraf.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.iraf.fold import annotate_audio_save_data
from ltx_trainer.iraf.gate import (
    adaptive_fusion,
    duplex_loss_stub,
    gate_binary_loss,
    gate_demo,
    gate_logits_stub,
    reliability_gate,
)
from ltx_trainer.iraf.mock import evaluation_smoke
from ltx_trainer.iraf.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    experimental_protocol,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_ms_marco_musan,
    table2_instructs2s,
)

__all__ = [
    "IrafConfig",
    "adaptive_fusion",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "duplex_loss_stub",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "experimental_protocol",
    "framework_card",
    "gate_binary_loss",
    "gate_demo",
    "gate_logits_stub",
    "headline_results",
    "pipeline_demo",
    "pipeline_demo_export",
    "reliability_gate",
    "table1_ms_marco_musan",
    "table2_instructs2s",
]
