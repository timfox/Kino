"""DSTA adapter smoke."""

from __future__ import annotations

from typing import Any


def toy_dst_inputs() -> dict[str, float]:
    return {
        "dwconv_out": 0.2,
        "conv_t": 0.5,
        "conv_h": 0.3,
        "conv_w": 0.1,
        "identity_skip": 0.4,
    }


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.dsta.dst import dst_forward

    inp = toy_dst_inputs()
    out = dst_forward(**inp)
    return {"dst_out": round(out, 4), **{f"in_{k}": round(v, 4) for k, v in inp.items()}}
