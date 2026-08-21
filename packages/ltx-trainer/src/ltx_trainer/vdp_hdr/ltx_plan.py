"""LTX integration plan for VDP-HDR stage-1 (video bracket fine-tune).

Paper (arXiv:2605.11628): fine-tune SVD UNet on synthetic brackets with channel-concat
conditioning latent ``z_c``; no CFG. Gopex maps this to native LTX AV training:

1. **Data** — ``hdr_ingest`` + :func:`hdr_to_ldr_bracket` on scene-linear ``hdr_latent`` shards;
   store ``ldr_bracket`` sidecar or multi-frame manifest rows (N=5, monotonic EV).
2. **Train** — short LoRA on transformer with ``with_audio: false``, frozen VAE; condition on
   frame 0 latent / first bracket frame; target sequence = full bracket (motion-free loss).
3. **Infer** — ``two_stage_hq_kino.py`` or ``inference.py`` with bracket prompt; pass output
   frames to :func:`recover_hdr_from_ldr` + trained Fusion UNet.

Env hooks:
  ``GOPEX_VDP_HDR_BRACKET_FRAMES=5``
  ``GOPEX_VDP_HDR_FUSION_CKPT=/path/to/fusion.pt``
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LtxVdpHdrPlan:
    bracket_frames: int = 5
    gamma: float = 2.2
    ltx_lora_steps: int = 4000
    fusion_steps: int = 25000
    no_cfg: bool = True


def default_plan() -> LtxVdpHdrPlan:
    import os

    return LtxVdpHdrPlan(
        bracket_frames=int(os.environ.get("GOPEX_VDP_HDR_BRACKET_FRAMES", "5")),
    )
