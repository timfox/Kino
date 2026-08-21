"""DeltaCam: differential intrinsic camera modeling for video generation (arXiv:2605.25266).

Reference utilities for Δ-parameterized neural camera adaptors, CCM FiLM conditioning,
disentangled style extraction, EXIF matching, Plücker extrinsics, scene proxies, and metrics.
Full Wan-2.1 VDM training is external.
"""

from ltx_trainer.deltacam.adaptors import StyleToDeltaAdaptor
from ltx_trainer.deltacam.ccm import CameraConditioningModule, apply_ccm
from ltx_trainer.deltacam.config import DeltaCamConfig, IntrinsicRange
from ltx_trainer.deltacam.delta_params import (
    delta_at_time,
    dual_effect_ramp,
    lock_to_anchor_delta,
    normalize_absolute,
    single_effect_ramp,
    trajectory_to_delta,
)
from ltx_trainer.deltacam.effect_attention import EffectAttentionBlock, concat_delta_plucker_cond
from ltx_trainer.deltacam.exif import ExifMetadataTokenizer, exif_sequence_embedding, normalize_metadata_vector
from ltx_trainer.deltacam.integration import encode_style_trajectory
from ltx_trainer.deltacam.layout import conditioning_layout, paper_limitations
from ltx_trainer.deltacam.film import DeltaFiLM, cascade_film, film_modulate
from ltx_trainer.deltacam.metrics import (
    reference_metrics_placeholder,
    table_wclip_sensitivity,
    wclip,
    wclip1,
    wclip10,
    wclip5,
)
from ltx_trainer.deltacam.pipeline import (
    camera_intrinsic_vocabulary,
    curriculum_stages,
    dataset_manifest,
    evaluation_demo,
    framework_card,
    intrinsic_groups,
    intrinsic_param_group,
    style_extraction_loss_weights,
    table_ablation,
    table_related_work,
    table_single_effect_control,
    table_style_extraction,
    training_step_demo,
)
from ltx_trainer.deltacam.plucker import plucker_from_rays, plucker_map_pinhole, zeros_plucker_map
from ltx_trainer.deltacam.proxies import SceneProxyMaps, assert_proxy_shapes, mask_proxy_streams
from ltx_trainer.deltacam.style import (
    DisentangledStyleEmbedder,
    StyleTrajectoryHead,
    mutual_information_penalty,
    normalized_cross_correlation,
    style_extraction_loss,
)
from ltx_trainer.deltacam.temporal import TemporalStyleEncoder

__all__ = [
    "CameraConditioningModule",
    "EffectAttentionBlock",
    "DeltaCamConfig",
    "DeltaFiLM",
    "DisentangledStyleEmbedder",
    "ExifMetadataTokenizer",
    "IntrinsicRange",
    "SceneProxyMaps",
    "StyleToDeltaAdaptor",
    "StyleTrajectoryHead",
    "TemporalStyleEncoder",
    "apply_ccm",
    "assert_proxy_shapes",
    "cascade_film",
    "delta_at_time",
    "dual_effect_ramp",
    "encode_style_trajectory",
    "evaluation_demo",
    "camera_intrinsic_vocabulary",
    "concat_delta_plucker_cond",
    "curriculum_stages",
    "exif_sequence_embedding",
    "film_modulate",
    "framework_card",
    "intrinsic_groups",
    "intrinsic_param_group",
    "lock_to_anchor_delta",
    "mask_proxy_streams",
    "mutual_information_penalty",
    "normalize_absolute",
    "normalize_metadata_vector",
    "normalized_cross_correlation",
    "conditioning_layout",
    "dataset_manifest",
    "paper_limitations",
    "plucker_from_rays",
    "plucker_map_pinhole",
    "reference_metrics_placeholder",
    "single_effect_ramp",
    "style_extraction_loss",
    "style_extraction_loss_weights",
    "table_ablation",
    "table_related_work",
    "table_single_effect_control",
    "table_style_extraction",
    "table_wclip_sensitivity",
    "training_step_demo",
    "trajectory_to_delta",
    "wclip",
    "wclip1",
    "wclip10",
    "wclip5",
    "zeros_plucker_map",
]
