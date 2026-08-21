"""nnAudio 2 PyTorch audio feature toolbox modernization (Roy et al., arXiv:2606.05394)."""

from ltx_trainer.nnaudio2.config import NnAudio2Config
from ltx_trainer.nnaudio2.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.nnaudio2.icqt import (
    estimate_frame_bound,
    icqt_snr_meets_target,
    landweber_contraction,
    landweber_reconstruct,
    landweber_step_size,
    reconstruction_snr_db,
)
from ltx_trainer.nnaudio2.mock import evaluation_smoke
from ltx_trainer.nnaudio2.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_fixes,
    pipeline_demo,
    table1_issue_fix_pairs,
    table2_regression_status,
)
from ltx_trainer.nnaudio2.scipy_compat import blackmanharris_import_path, cfp_window_available
from ltx_trainer.nnaudio2.stft import (
    IstftFreqScaleError,
    UNSUPPORTED_ISTFT_FREQ_SCALES,
    overlap_add_length,
    round_trip_error_uniform,
    torchscript_fixes,
    validate_istft_freq_scale,
)
from ltx_trainer.nnaudio2.vqt_cqt import vqt_cqt_max_diff, vqt_routes_to_cqt

__all__ = [
    "IstftFreqScaleError",
    "NnAudio2Config",
    "UNSUPPORTED_ISTFT_FREQ_SCALES",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "blackmanharris_import_path",
    "cfp_window_available",
    "eval_smoke",
    "estimate_frame_bound",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_fixes",
    "icqt_snr_meets_target",
    "landweber_contraction",
    "landweber_reconstruct",
    "landweber_step_size",
    "overlap_add_length",
    "pipeline_demo",
    "pipeline_demo_export",
    "reconstruction_snr_db",
    "round_trip_error_uniform",
    "table1_issue_fix_pairs",
    "table2_regression_status",
    "torchscript_fixes",
    "validate_istft_freq_scale",
    "vqt_cqt_max_diff",
    "vqt_routes_to_cqt",
]

from ltx_trainer.nnaudio2.fold import annotate_audio_save_data  # noqa: E402
