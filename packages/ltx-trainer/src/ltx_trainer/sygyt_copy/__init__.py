"""Sygyt articulatory copy-synthesis (Cámara et al., arXiv:2606.04943 / DAFx26)."""

from ltx_trainer.sygyt_copy.config import SygytCopyConfig
from ltx_trainer.sygyt_copy.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.sygyt_copy.fold import annotate_audio_save_data
from ltx_trainer.sygyt_copy.losses import combined_loss, overtone_salience, overtone_salience_loss
from ltx_trainer.sygyt_copy.metrics import formant_peak_error_hz, lsd_reduction_pct
from ltx_trainer.sygyt_copy.mock import evaluation_smoke
from ltx_trainer.sygyt_copy.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table2_copy_synthesis,
    table3_overtone_errors,
    table4_ablation,
)
from ltx_trainer.sygyt_copy.waveguide import (
    apply_damping,
    clip_damping,
    reflection_coefficient,
    three_way_scattering,
)

__all__ = [
    "SygytCopyConfig",
    "annotate_audio_save_data",
    "apply_damping",
    "benchmarks_bundle",
    "clip_damping",
    "combined_loss",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "formant_peak_error_hz",
    "framework_card",
    "lsd_reduction_pct",
    "overtone_salience",
    "overtone_salience_loss",
    "pipeline_demo",
    "pipeline_demo_export",
    "reflection_coefficient",
    "table2_copy_synthesis",
    "table3_overtone_errors",
    "table4_ablation",
    "three_way_scattering",
]
