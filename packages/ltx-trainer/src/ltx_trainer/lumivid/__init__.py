"""LumiVid: LogC3-aligned HDR video from SDR via frozen VAE + LoRA (arXiv:2604.11788)."""

from ltx_trainer.lumivid.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.lumivid.logc3_codec import scene_linear_to_vae_pixels, vae_pixels_to_scene_linear
from ltx_trainer.lumivid.model import LumiVid, LumiVidConfig
from ltx_trainer.lumivid.paper import evaluation_demo, framework_card

__all__ = [
    "LumiVid",
    "LumiVidConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "scene_linear_to_vae_pixels",
    "vae_pixels_to_scene_linear",
]
