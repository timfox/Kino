"""Training-free QPA encode demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvc_erp_qpa.config import NvcErpQpaConfig
from ltx_trainer.nvc_erp_qpa.nvc_stub import DcvcRtQpaStub
from ltx_trainer.nvc_erp_qpa.quality_parameter import mean_delta_q
from ltx_trainer.nvc_erp_qpa.synthetic import synthetic_erp_frame
from ltx_trainer.nvc_erp_qpa.ws_psnr import ws_psnr_db, yuv_ws_psnr_proxy


def encode_step(cfg: NvcErpQpaConfig | None = None) -> dict[str, float]:
    cfg = cfg or NvcErpQpaConfig()
    frame = synthetic_erp_frame(cfg)
    model = DcvcRtQpaStub(cfg)
    out = model(frame)
    psnr_qpa = yuv_ws_psnr_proxy(frame, out["recon_qpa"])
    psnr_base = yuv_ws_psnr_proxy(frame, out["recon_baseline"])
    return {
        "ws_psnr_qpa_db": psnr_qpa,
        "ws_psnr_baseline_db": psnr_base,
        "q_map_min": float(out["q_map"].min()),
        "q_map_max": float(out["q_map"].max()),
        "mean_delta_q": mean_delta_q(
            q_num=cfg.q_num, lambda_min=cfg.lambda_min, lambda_max=cfg.lambda_max
        ),
    }


def interpolation_ablation(cfg: NvcErpQpaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NvcErpQpaConfig()
    frame = synthetic_erp_frame(cfg)
    with_interp = DcvcRtQpaStub(cfg)
    cfg_floor = NvcErpQpaConfig(**{**cfg.__dict__, "use_interpolation": False})
    no_interp = DcvcRtQpaStub(cfg_floor)
    o1 = with_interp(frame)
    o2 = no_interp(frame)
    return {
        "interp_ws_psnr": yuv_ws_psnr_proxy(frame, o1["recon_qpa"]),
        "floor_ws_psnr": yuv_ws_psnr_proxy(frame, o2["recon_qpa"]),
        "q_map_span": float(o1["q_map"].max() - o1["q_map"].min()),
    }


def evaluation_demo_run(cfg: NvcErpQpaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NvcErpQpaConfig()
    metrics = encode_step(cfg)
    abl = interpolation_ablation(cfg)
    return {"encode": metrics, "interpolation_ablation": abl}
