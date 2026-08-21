"""F-RNG: Feed-Forward Relightable Neural Gaussians (Fu et al., arXiv:2605.25975).

Trainable small networks around **frozen** large priors (RelitLRM, DiffusionRenderer): saliency-driven
fine geometry synthesis, MaterialFormer with light-independence loss, and a universal neural appearance
decoder. This package does **not** ship or run RelitLRM / DiffusionRenderer — pass their latents or use
synthetic tensors for development.
"""

from ltx_trainer.frng.config import FRNGConfig
from ltx_trainer.frng.decoder import UniversalNeuralAppearanceDecoder
from ltx_trainer.frng.geometry import (
    FineGeoDetokenizer,
    FineGeometrySynthesis,
    PriorTokenizer,
    bilinear_corner_aggregate,
    grid_to_tokens,
    tokens_to_grid,
)
from ltx_trainer.frng.losses import PerceptualLossPlaceholder, frng_composite_loss, l2_rendering_loss
from ltx_trainer.frng.material_former import MatDetokenizer, MaterialFormer, light_independence_loss
from ltx_trainer.frng.pipeline import FRNGComplete, FRNGTrainableStack, FRNGViews, rgb_to_gray
from ltx_trainer.frng.saliency import (
    patch_grid_shape,
    raw_texture_saliency,
    saliency_map,
    saliency_patch_scores,
    topk_patch_flat_indices,
)

__all__ = [
    "FRNGComplete",
    "FRNGConfig",
    "FRNGTrainableStack",
    "FRNGViews",
    "FineGeoDetokenizer",
    "FineGeometrySynthesis",
    "MatDetokenizer",
    "MaterialFormer",
    "PerceptualLossPlaceholder",
    "PriorTokenizer",
    "UniversalNeuralAppearanceDecoder",
    "bilinear_corner_aggregate",
    "frng_composite_loss",
    "grid_to_tokens",
    "l2_rendering_loss",
    "light_independence_loss",
    "patch_grid_shape",
    "raw_texture_saliency",
    "rgb_to_gray",
    "saliency_map",
    "saliency_patch_scores",
    "tokens_to_grid",
    "topk_patch_flat_indices",
]
