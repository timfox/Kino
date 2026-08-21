"""QAT lightweight neural distinguisher (Xiong et al., arXiv:2603.05791)."""

from ltx_trainer.nd_qat.architecture import (
    conv0_gohr_ops,
    conv0_lightweight_ops,
    table1_conv0_weights,
    table5_conv0_boolean,
)
from ltx_trainer.nd_qat.boolean_ops import BooleanConv2d, BooleanLinear, indicator
from ltx_trainer.nd_qat.config import NDQATConfig, PAPER_DOI, PAPER_TITLE, PAPER_URL, QUANT_BITS_EQUIVALENT
from ltx_trainer.nd_qat.lsq import lsq_quantize, quantize_entropy_bits, ternary_project
from ltx_trainer.nd_qat.metrics import ops_reduction_ratio, table4_comparison, table6_conv0_only
from ltx_trainer.nd_qat.model import GohrDistinguisher, LightweightDistinguisher
from ltx_trainer.nd_qat.pipeline import count_parameters, knowledge, paper_report, train_step
from ltx_trainer.nd_qat.synthetic import ciphertext_to_feature, dataset_summary, speck32_encrypt_pair, synthetic_batch

__all__ = [
    "BooleanConv2d",
    "BooleanLinear",
    "GohrDistinguisher",
    "LightweightDistinguisher",
    "NDQATConfig",
    "PAPER_DOI",
    "PAPER_TITLE",
    "PAPER_URL",
    "QUANT_BITS_EQUIVALENT",
    "ciphertext_to_feature",
    "conv0_gohr_ops",
    "conv0_lightweight_ops",
    "count_parameters",
    "dataset_summary",
    "indicator",
    "knowledge",
    "lsq_quantize",
    "ops_reduction_ratio",
    "paper_report",
    "quantize_entropy_bits",
    "speck32_encrypt_pair",
    "synthetic_batch",
    "table1_conv0_weights",
    "table4_comparison",
    "table5_conv0_boolean",
    "table6_conv0_only",
    "ternary_project",
    "train_step",
]
