"""Sharp Picture: transformer generalization via Fourier spectra (Lintilhac et al., arXiv:2605.20988)."""

from ltx_trainer.sharp_picture.config import SharpPictureConfig
from ltx_trainer.sharp_picture.fourier import (
    fourier_expansion,
    parity_characteristic,
    random_positive_coefficients,
)
from ltx_trainer.sharp_picture.metrics import (
    cot_bound,
    gu_unperturbed,
    norm_proxy_l,
    onepass_bound,
    semi_analytic_bound,
)
from ltx_trainer.sharp_picture.pipeline import (
    evaluation_demo,
    framework_card,
    parity_cot_comparison,
    table_domination_assumption,
    table_empirical_gap_reference,
    table_semi_analytic_bounds,
)

__all__ = [
    "SharpPictureConfig",
    "cot_bound",
    "fourier_expansion",
    "framework_card",
    "gu_unperturbed",
    "norm_proxy_l",
    "onepass_bound",
    "parity_characteristic",
    "parity_cot_comparison",
    "random_positive_coefficients",
    "semi_analytic_bound",
    "table_domination_assumption",
    "table_empirical_gap_reference",
    "table_semi_analytic_bounds",
    "evaluation_demo",
]