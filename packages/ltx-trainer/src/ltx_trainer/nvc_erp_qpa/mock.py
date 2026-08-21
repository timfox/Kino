"""CPU smoke for NVC ERP QPA stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvc_erp_qpa.benchmarks import benchmarks_bundle
from ltx_trainer.nvc_erp_qpa.config import PAPER_ARXIV, NvcErpQpaConfig
from ltx_trainer.nvc_erp_qpa.datasets import datasets_card
from ltx_trainer.nvc_erp_qpa.nvc_stub import DcvcRtQpaStub
from ltx_trainer.nvc_erp_qpa.paper import framework_card
from ltx_trainer.nvc_erp_qpa.pipeline import encode_step, evaluation_demo_run
from ltx_trainer.nvc_erp_qpa.quality_parameter import adaptive_q_tilde, latitude_from_row
from ltx_trainer.nvc_erp_qpa.synthetic import synthetic_erp_frame
from ltx_trainer.nvc_erp_qpa.vector_bank import interpolate_vector


def evaluation_smoke() -> dict[str, Any]:
    cfg = NvcErpQpaConfig()
    frame = synthetic_erp_frame(cfg)
    model = DcvcRtQpaStub(cfg)
    out = model(frame)
    bundle = benchmarks_bundle()
    metrics = encode_step(cfg)

    # Equator vs pole: pole should have lower q̃ (higher latitude magnitude)
    q_eq = float(
        adaptive_q_tilde(
            cfg.q0,
            latitude_from_row(cfg.erp_height // 2, cfg.erp_height),
            q_num=cfg.q_num,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
        )
    )
    q_pole = float(
        adaptive_q_tilde(
            cfg.q0,
            latitude_from_row(0, cfg.erp_height),
            q_num=cfg.q_num,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
        )
    )

    v_mid = interpolate_vector(model.mod_e.bank, float(cfg.q0))

    return {
        "package": "nvc_erp_qpa",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "recon_shape": list(out["recon_qpa"].shape),
        "q_equator": q_eq,
        "q_pole": q_pole,
        "pole_lower_q_than_equator": q_pole < q_eq,
        "interp_vector_dim": list(v_mid.shape),
        "bd_rate_avg_pct": bundle["table1_bd_rate_proposed"]["Average"],
        "encode_metrics": metrics,
        "demo": evaluation_demo_run(cfg),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
    }
