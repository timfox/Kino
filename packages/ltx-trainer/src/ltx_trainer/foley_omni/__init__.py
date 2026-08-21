"""Foley-Omni — unified V2ST + task-level audio generation (arXiv:2606.03672)."""

from ltx_trainer.foley_omni.conditioning import (
    apply_sync_injection,
    concat_unified_context,
    format_structured_prompt,
    parse_structured_fields,
    sync_adapter,
    toy_project_conditions,
)
from ltx_trainer.foley_omni.config import FoleyOmniConfig
from ltx_trainer.foley_omni.curation import (
    ClipFilterThresholds,
    bandit_stem_passes,
    passes_visual_audio_filters,
    rms_energy_db,
    verify_component_labels,
)
from ltx_trainer.foley_omni.layout import LIMITATIONS
from ltx_trainer.foley_omni.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_summary,
    pipeline_demo,
    table_ablation,
    table_text_tasks,
    table_v2st_main,
    table_vgg_sound,
    training_data_groups,
    v2st_bench_composition,
)
from ltx_trainer.foley_omni.training import (
    CurriculumStage,
    curriculum_schedule,
    flow_matching_loss,
    interpolate_path,
    target_velocity,
    training_step_smoke,
)

__all__ = [
    "ClipFilterThresholds",
    "CurriculumStage",
    "FoleyOmniConfig",
    "LIMITATIONS",
    "apply_sync_injection",
    "bandit_stem_passes",
    "benchmarks_bundle",
    "concat_unified_context",
    "curriculum_schedule",
    "evaluation_demo",
    "flow_matching_loss",
    "format_structured_prompt",
    "framework_card",
    "interpolate_path",
    "knowledge_summary",
    "parse_structured_fields",
    "passes_visual_audio_filters",
    "pipeline_demo",
    "rms_energy_db",
    "sync_adapter",
    "table_ablation",
    "table_text_tasks",
    "table_v2st_main",
    "table_vgg_sound",
    "target_velocity",
    "toy_project_conditions",
    "training_data_groups",
    "training_step_smoke",
    "verify_component_labels",
    "v2st_bench_composition",
]
