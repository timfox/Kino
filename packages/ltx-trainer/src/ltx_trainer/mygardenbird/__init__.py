"""MyGardenBird — ML-ready Malaysian bird sound dataset (arXiv:2606.06975)."""

from ltx_trainer.mygardenbird.config import MygardenbirdConfig
from ltx_trainer.mygardenbird.eval import eval_smoke, pipeline_demo
from ltx_trainer.mygardenbird.fold import annotate_audio_save_data
from ltx_trainer.mygardenbird.mock import evaluation_smoke
from ltx_trainer.mygardenbird.pipeline import (
    benchmarks_bundle,
    curation_protocol,
    evaluation_demo,
    framework_card,
    headline_results,
    metadata_schema,
    table2_species_composition,
    table3_mip_splits,
    table6_birdnet_validation,
    table7_cnn_accuracy,
    table8_augmentation_16k,
)
from ltx_trainer.mygardenbird.snr import estimate_snr_db, snr_demo
from ltx_trainer.mygardenbird.species import SPECIES, clip_id, species_demo, species_table, verify_balance
from ltx_trainer.mygardenbird.split import greedy_source_split, split_demo, verify_no_source_leakage
from ltx_trainer.mygardenbird.spectrogram import mel_params, spectrogram_demo

__all__ = [
    "MygardenbirdConfig",
    "SPECIES",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "clip_id",
    "curation_protocol",
    "estimate_snr_db",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "greedy_source_split",
    "headline_results",
    "mel_params",
    "metadata_schema",
    "pipeline_demo",
    "snr_demo",
    "species_demo",
    "species_table",
    "spectrogram_demo",
    "split_demo",
    "table2_species_composition",
    "table3_mip_splits",
    "table6_birdnet_validation",
    "table7_cnn_accuracy",
    "table8_augmentation_16k",
    "verify_balance",
    "verify_no_source_leakage",
]
