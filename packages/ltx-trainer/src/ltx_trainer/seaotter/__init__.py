"""SEAOTTER sensor-embedded autoencoder + one-time JPEG transcode (arXiv:2606.03940)."""

from ltx_trainer.seaotter.color_transform import LearnedColorTransform
from ltx_trainer.seaotter.config import SeaotterConfig
from ltx_trainer.seaotter.frappe import FrappeDecoder, FrappeEncoder, frappe_pipeline_smoke
from ltx_trainer.seaotter.jpeg_sandwich import JpegSandwich, multi_rate_loss, sandwich_training_smoke
from ltx_trainer.seaotter.ltx_plan import ltx_integration_plan
from ltx_trainer.seaotter.pipeline import (
    evaluation_demo,
    framework_card,
    table1_matched_rate_summary,
    table_deployment_tiers,
    table_standalone_kodak,
)
from ltx_trainer.seaotter.quantization import LearnedQuantizationBank, quantize_matrix
from ltx_trainer.seaotter.rate_proxy import rate_proxy_bpp
from ltx_trainer.seaotter.transcode import TranscodeResult, seaotter_forward, transcode_smoke

__all__ = [
    "FrappeDecoder",
    "FrappeEncoder",
    "JpegSandwich",
    "LearnedColorTransform",
    "LearnedQuantizationBank",
    "SeaotterConfig",
    "TranscodeResult",
    "evaluation_demo",
    "framework_card",
    "frappe_pipeline_smoke",
    "ltx_integration_plan",
    "multi_rate_loss",
    "quantize_matrix",
    "rate_proxy_bpp",
    "sandwich_training_smoke",
    "seaotter_forward",
    "table1_matched_rate_summary",
    "table_deployment_tiers",
    "table_standalone_kodak",
    "transcode_smoke",
]
