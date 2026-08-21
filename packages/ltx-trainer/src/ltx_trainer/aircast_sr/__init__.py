"""AirCast-SR: km-scale atmospheric super-resolution via LCM diffusion (arXiv:2605.26130)."""

from ltx_trainer.aircast_sr.config import AirCastSRConfig
from ltx_trainer.aircast_sr.interpolate import trilinear_time_spatial
from ltx_trainer.aircast_sr.lcm import fresh_denoiser, lcm_sample, training_mse_loss
from ltx_trainer.aircast_sr.layout import LIMITATIONS
from ltx_trainer.aircast_sr.metrics import bias, mae, pearson_r, rmse, skill_row
from ltx_trainer.aircast_sr.mock import evaluation_smoke
from ltx_trainer.aircast_sr.normalize import normalize_target_stack
from ltx_trainer.aircast_sr.patches import extract_patches, merge_patches
from ltx_trainer.aircast_sr.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
)
from ltx_trainer.aircast_sr.psd import radial_power_spectral_density
from ltx_trainer.aircast_sr.schedule import add_noise, diffusion_schedule, lcm_inference_timesteps
from ltx_trainer.aircast_sr.tables import (
    table_i_precipitation,
    table_ii_temperature,
    table_iii_surface_pressure,
    table_iv_humidity,
    table_v_longwave,
    table_vi_u_wind,
    table_vii_v_wind,
    table_viii_zero_shot_t2m,
)
from ltx_trainer.aircast_sr.unet3d import UNet3DWeights, build_toy_unet3d, unet3d_forward
from ltx_trainer.aircast_sr.variables import CONDITIONING_CHANNELS, TARGET_VARIABLES, channel_spec

__all__ = [
    "AirCastSRConfig",
    "CONDITIONING_CHANNELS",
    "LIMITATIONS",
    "TARGET_VARIABLES",
    "UNet3DWeights",
    "add_noise",
    "benchmarks_bundle",
    "bias",
    "build_toy_unet3d",
    "channel_spec",
    "diffusion_schedule",
    "evaluation_demo",
    "evaluation_smoke",
    "extract_patches",
    "framework_card",
    "fresh_denoiser",
    "headline_results",
    "lcm_inference_timesteps",
    "lcm_sample",
    "mae",
    "merge_patches",
    "normalize_target_stack",
    "pearson_r",
    "radial_power_spectral_density",
    "rmse",
    "skill_row",
    "table_i_precipitation",
    "table_ii_temperature",
    "table_iii_surface_pressure",
    "table_iv_humidity",
    "table_v_longwave",
    "table_vi_u_wind",
    "table_vii_v_wind",
    "table_viii_zero_shot_t2m",
    "training_mse_loss",
    "trilinear_time_spatial",
    "unet3d_forward",
]
