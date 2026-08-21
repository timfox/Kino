"""LC-DeepBeam: linearly constrained deep beamformer stubs (arXiv:2605.21141)."""

from ltx_trainer.lc_deepbeam.config import LcDeepBeamConfig
from ltx_trainer.lc_deepbeam.cw_rtf import (
    interference_subspace_from_eigs,
    inv_sqrtm_hermitian,
    noise_covariance,
    rtf_from_dominant_eig,
    whiten,
    whitened_covariance,
)
from ltx_trainer.lc_deepbeam.lcmv import lcmv_weights, wideband_beampower
from ltx_trainer.lc_deepbeam.layout import LIMITATIONS
from ltx_trainer.lc_deepbeam.losses import (
    null_penalty_db,
    pass_penalty,
    schedule_lambdas,
    si_sdr,
    total_loss,
)
from ltx_trainer.lc_deepbeam.mock import evaluation_smoke
from ltx_trainer.lc_deepbeam.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_three_speaker_anechoic,
    table2_two_speaker_reverberant,
    table3_fully_overlapped_conceptual,
)

__all__ = [
    "LIMITATIONS",
    "LcDeepBeamConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "interference_subspace_from_eigs",
    "inv_sqrtm_hermitian",
    "lcmv_weights",
    "noise_covariance",
    "null_penalty_db",
    "pass_penalty",
    "rtf_from_dominant_eig",
    "schedule_lambdas",
    "si_sdr",
    "table1_three_speaker_anechoic",
    "table2_two_speaker_reverberant",
    "table3_fully_overlapped_conceptual",
    "total_loss",
    "wideband_beampower",
    "whiten",
    "whitened_covariance",
]

