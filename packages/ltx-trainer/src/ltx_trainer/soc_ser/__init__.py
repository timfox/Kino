"""SOC — second-order correlation SER (arXiv:2606.06550)."""

from ltx_trainer.soc_ser.config import SocSerConfig
from ltx_trainer.soc_ser.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.soc_ser.fold import annotate_audio_save_data
from ltx_trainer.soc_ser.layer import (
    log_euclidean_map,
    soc_demo,
    soc_layer,
    subspace_dim_curve,
    trace_normalized_covariance,
    vech,
)
from ltx_trainer.soc_ser.mock import evaluation_smoke
from ltx_trainer.soc_ser.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    experimental_protocol,
    framework_card,
    headline_results,
    lem_ablation_delta,
    pipeline_demo,
    table1_esd_ravdess,
)

__all__ = [
    "SocSerConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "experimental_protocol",
    "framework_card",
    "headline_results",
    "lem_ablation_delta",
    "log_euclidean_map",
    "pipeline_demo",
    "pipeline_demo_export",
    "soc_demo",
    "soc_layer",
    "subspace_dim_curve",
    "table1_esd_ravdess",
    "trace_normalized_covariance",
    "vech",
]
