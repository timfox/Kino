"""LOPEO — robust evaluation of stimulus reconstruction AAD on unbalanced EEG datasets."""

from ltx_trainer.lopeo.balance import balance_index, counts_from_trials
from ltx_trainer.lopeo.config import LopeoConfig
from ltx_trainer.lopeo.datasets import (
    construct_dtu_trials,
    construct_kul_trials,
    construct_nju_ceegrid_trials,
    dataset_condition_summary,
)
from ltx_trainer.lopeo.cv import (
    collect_stimulus_pairs,
    pair_leakage_count,
    partition_trials_loeo,
    partition_trials_loto,
    partition_trials_lopeo,
    trial_stimulus_pair,
    unordered_pair,
)
from ltx_trainer.lopeo.layout import LIMITATIONS
from ltx_trainer.lopeo.mock import IdentityMemorizingDecoder, envelope_reconstruction_demo
from ltx_trainer.lopeo.stats import lopeo_mitigation_summary, significance_table
from ltx_trainer.lopeo.metrics import (
    contrastive_pcc_loss,
    decoding_accuracy,
    pcc_loss,
    pearson_corr,
    rho_delta,
)
from ltx_trainer.lopeo.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    filter_table_ii,
    framework_card,
    headline_results,
    pipeline_demo,
    table_i_datasets,
    table_ii_results,
)

__all__ = [
    "LIMITATIONS",
    "IdentityMemorizingDecoder",
    "LopeoConfig",
    "balance_index",
    "benchmarks_bundle",
    "collect_stimulus_pairs",
    "construct_dtu_trials",
    "construct_kul_trials",
    "construct_nju_ceegrid_trials",
    "contrastive_pcc_loss",
    "counts_from_trials",
    "dataset_condition_summary",
    "decoding_accuracy",
    "envelope_reconstruction_demo",
    "evaluation_demo",
    "filter_table_ii",
    "framework_card",
    "headline_results",
    "lopeo_mitigation_summary",
    "significance_table",
    "pair_leakage_count",
    "partition_trials_loeo",
    "partition_trials_loto",
    "partition_trials_lopeo",
    "pcc_loss",
    "pearson_corr",
    "pipeline_demo",
    "rho_delta",
    "table_i_datasets",
    "table_ii_results",
    "trial_stimulus_pair",
    "unordered_pair",
]
