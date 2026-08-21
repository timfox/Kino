"""VDP-HDR: single-shot HDR via video diffusion prior (Talegaonkar et al. arXiv:2605.11628).

Stage 1 (optional LTX fine-tune): motion-free LDR exposure bracket conditioned on one LDR frame.
Stage 2 (default trainable here): Fusion UNet predicts per-pixel weights; fuse in PU-21 space.

See :mod:`ltx_trainer.vdp_hdr.pipeline` and ``./scripts/kino-vdp-hdr.sh``.
"""

from ltx_trainer.vdp_hdr.bracket import BracketConfig, hdr_to_ldr_bracket, ldr_bracket_from_single
from ltx_trainer.vdp_hdr.fusion import FusionUNet, bracket_to_linear, fuse_bracket, mertens_fusion
from ltx_trainer.vdp_hdr.model import VdpHdr, VdpHdrConfig
from ltx_trainer.vdp_hdr.pipeline import load_vdp_hdr_checkpoint, recover_hdr_from_ldr

__all__ = [
    "BracketConfig",
    "FusionUNet",
    "VdpHdr",
    "VdpHdrConfig",
    "bracket_to_linear",
    "fuse_bracket",
    "hdr_to_ldr_bracket",
    "ldr_bracket_from_single",
    "load_vdp_hdr_checkpoint",
    "mertens_fusion",
    "recover_hdr_from_ldr",
]
