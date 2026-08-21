"""Multi-probe audit of clinical-interview depression detection benchmarks."""

from ltx_trainer.dep_audit.config import DepAuditConfig
from ltx_trainer.dep_audit.layout import LIMITATIONS
from ltx_trainer.dep_audit.metrics import macro_f1_from_confusion, paired_shift
from ltx_trainer.dep_audit.mock import evaluation_smoke
from ltx_trainer.dep_audit.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_iii_loso,
    table_iv_official_instability,
    table_v_cmdc_external,
    table_vi_androids_external,
    table_vii_topic_stress,
)
from ltx_trainer.dep_audit.probes import table_i_probes
from ltx_trainer.dep_audit.srds import classify_chunk_stub, heavy_neutral_delta, topic_score_rubric

__all__ = [
    "LIMITATIONS",
    "DepAuditConfig",
    "benchmarks_bundle",
    "classify_chunk_stub",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "heavy_neutral_delta",
    "macro_f1_from_confusion",
    "paired_shift",
    "pipeline_demo",
    "table_i_probes",
    "table_iii_loso",
    "table_iv_official_instability",
    "table_v_cmdc_external",
    "table_vi_androids_external",
    "table_vii_topic_stress",
    "topic_score_rubric",
]
