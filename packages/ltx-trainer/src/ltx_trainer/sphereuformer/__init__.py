"""SphereUFormer — spherical U-Transformer for 360° depth and segmentation."""

from ltx_trainer.sphereuformer.benchmarks import benchmarks_bundle
from ltx_trainer.sphereuformer.config import PAPER_ARXIV, PAPER_TITLE, SphereUFormerConfig
from ltx_trainer.sphereuformer.sphereuformer_net import SphereUFormerStub

__all__ = [
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "SphereUFormerConfig",
    "SphereUFormerStub",
    "benchmarks_bundle",
]
