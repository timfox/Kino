"""BiEAR — adaptive binaural front-end (arXiv:2606.06795)."""

from ltx_trainer.biear.binaural import (
    binaural_demo,
    binaural_embeddings,
    compute_cc,
    compute_ild,
    compute_ipd,
)
from ltx_trainer.biear.config import BiearConfig
from ltx_trainer.biear.controller import controller_demo, modulate_q_frame
from ltx_trainer.biear.eval import eval_smoke, pipeline_demo
from ltx_trainer.biear.filterbank import apply_q_control, filterbank_demo
from ltx_trainer.biear.fold import annotate_audio_save_data
from ltx_trainer.biear.mock import evaluation_smoke
from ltx_trainer.biear.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_brir_datasets,
    table2_anechoic,
    table3_real_rooms,
)
from ltx_trainer.biear.sad_net import sad_net_demo, sad_net_forward, sector_for_azimuth

__all__ = [
    "BiearConfig",
    "annotate_audio_save_data",
    "apply_q_control",
    "benchmarks_bundle",
    "binaural_demo",
    "binaural_embeddings",
    "compute_cc",
    "compute_ild",
    "compute_ipd",
    "controller_demo",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "filterbank_demo",
    "framework_card",
    "headline_results",
    "modulate_q_frame",
    "pipeline_demo",
    "sad_net_demo",
    "sad_net_forward",
    "sector_for_azimuth",
    "table1_brir_datasets",
    "table2_anechoic",
    "table3_real_rooms",
]
