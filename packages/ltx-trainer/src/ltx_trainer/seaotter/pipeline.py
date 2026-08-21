"""Framework card, Table 1, deployment tiers, evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.seaotter.config import SeaotterConfig
from ltx_trainer.seaotter.frappe import frappe_pipeline_smoke, sensor_encode_throughput_mpx_s
from ltx_trainer.seaotter.jpeg_sandwich import sandwich_training_smoke
from ltx_trainer.seaotter.ltx_plan import ltx_integration_plan
from ltx_trainer.seaotter.transcode import compression_ratio, transcode_smoke


def framework_card(cfg: SeaotterConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeaotterConfig()
    return {
        "name": "SEAOTTER",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "repo": cfg.repo_url,
        "pipeline": "x̂ = F⁻¹ ∘ J_Q ∘ F ∘ G_S ∘ C ∘ G_A(x)",
        "stages": [
            "Sensor: frozen FRAPPE encoder G_A (10–100 MAC/pixel)",
            "Uplink: lossless int8 latent (JPEG-LS)",
            "Cloud: fine-tuned FRAPPE decoder G_S + one-time JPEG transcode",
            "Consumer: vanilla JPEG decode + optional F⁻¹ (~81 MAC/pixel)",
        ],
        "headline_vs_avif_cr200": {
            "encode_speedup": "7×",
            "decode_speedup": "3.5×",
            "imagenet_top1_delta_pp": "+8%",
        },
        "jpeg_sandwich": "Learned 3×3 color transform + K DCT Q matrices (de novo)",
        "variants": ("SEAOTTER-ZS", "SEAOTTER-FT", "WaLLoC-SEAOTTER-ZS", "FRAPPE-SEAOTTER-FT"),
    }


def table1_matched_rate_summary() -> list[dict[str, Any]]:
    """Table 1 — matched transmit-bpp operating points (§3)."""
    return [
        {
            "task": "ImageNet-1k (384²) cls",
            "resolution": "384x384",
            "metric": "top-1 (%)",
            "rows": [
                {"pipeline": "AVIF", "op": "q=1", "transmit_cr": 165, "storage_cr": 165, "score": 61.15, "encode_mpx_s": 5.51, "decode_mpx_s": 19.75},
                {"pipeline": "AVIF (max-speed)", "op": "q=1", "transmit_cr": 154, "storage_cr": 154, "score": 61.02, "encode_mpx_s": 25.73, "decode_mpx_s": 19.53},
                {"pipeline": "FRAPPE", "op": "n=12", "transmit_cr": 221, "storage_cr": 221, "score": 56.22, "encode_mpx_s": 177.76, "decode_mpx_s": 0.68},
                {"pipeline": "WaLLoC", "op": "p=16", "transmit_cr": 167, "storage_cr": 167, "score": 60.98, "encode_mpx_s": 30.17, "decode_mpx_s": 3.94},
                {"pipeline": "SEAOTTER-ZS", "op": "n=12", "transmit_cr": 221, "storage_cr": 19, "score": 60.25, "encode_mpx_s": 177.76, "decode_mpx_s": 65.35},
                {"pipeline": "SEAOTTER-FT", "op": "n=12", "transmit_cr": 221, "storage_cr": 27, "score": 69.02, "encode_mpx_s": 177.76, "decode_mpx_s": 67.97},
            ],
        },
        {
            "task": "ADE20K (512²) seg",
            "resolution": "512x512",
            "metric": "mIoU (%)",
            "rows": [
                {"pipeline": "AVIF", "op": "q=5", "transmit_cr": 279, "storage_cr": 279, "score": 32.75, "encode_mpx_s": 5.43, "decode_mpx_s": 19.79},
                {"pipeline": "FRAPPE", "op": "n=12", "transmit_cr": 256, "storage_cr": 256, "score": 29.09, "encode_mpx_s": 256.37, "decode_mpx_s": 0.68},
                {"pipeline": "SEAOTTER-FT", "op": "n=12", "transmit_cr": 256, "storage_cr": 46, "score": 32.77, "encode_mpx_s": 256.37, "decode_mpx_s": 67.97},
            ],
        },
        {
            "task": "ImageNet-1k (naflex) clip",
            "resolution": "naflex 256 tokens",
            "metric": "SigLIP top-1 (%)",
            "rows": [
                {"pipeline": "AVIF (max-speed)", "op": "q=1", "transmit_cr": 91, "storage_cr": 91, "score": 44.19, "encode_mpx_s": 19.45, "decode_mpx_s": 19.53},
                {"pipeline": "FRAPPE", "op": "n=12", "transmit_cr": 169, "storage_cr": 169, "score": 41.51, "encode_mpx_s": 96.74, "decode_mpx_s": 0.68},
                {"pipeline": "SEAOTTER-FT", "op": "n=12", "transmit_cr": 169, "storage_cr": 37, "score": 48.22, "encode_mpx_s": 96.74, "decode_mpx_s": 67.97},
            ],
        },
    ]


def table_deployment_tiers() -> list[dict[str, Any]]:
    """Table 8 excerpt — SEAOTTER-FT tier clearance at low n."""
    cfg = SeaotterConfig()
    rows = []
    for n, enc in [(3, 601.47), (6, 317.23), (9, 271.81), (12, 177.76)]:
        tbpp = {3: 0.0122, 6: 0.0380, 9: 0.0637, 12: 0.1086}[n]
        cr = compression_ratio(tbpp)
        rows.append(
            {
                "pipeline": "SEAOTTER-FT",
                "op": f"n={n}",
                "transmit_cr": round(cr, 1),
                "encode_mpx_s": enc,
                "ble": cr >= cfg.tier_ble_cr_min and enc >= cfg.tier_ble_encode_mpx_s,
                "5g": cr >= cfg.tier_5g_cr_min and enc >= cfg.tier_5g_encode_mpx_s,
                "wifi": cr >= cfg.tier_wifi_cr_min and enc >= cfg.tier_wifi_encode_mpx_s,
            }
        )
    return rows


def table_standalone_kodak() -> list[dict[str, float | str]]:
    """Table 2 — standalone learned JPEG on Kodak (Appendix A.3)."""
    return [
        {"codec": "ITU T.81 4:4:4", "setting": "q=53", "bpp": 1.103, "psnr_db": 32.90},
        {"codec": "SEAOTTER (ours)", "setting": "k=0", "bpp": 1.099, "psnr_db": 33.17},
        {"codec": "SEAOTTER (ours)", "setting": "k=1", "bpp": 1.909, "psnr_db": 37.77},
        {"codec": "SEAOTTER (ours)", "setting": "k=2", "bpp": 2.870, "psnr_db": 40.89},
    ]


def evaluation_demo(cfg: SeaotterConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeaotterConfig()
    return {
        "framework": framework_card(cfg),
        "ltx_plan": ltx_integration_plan(cfg),
        "frappe_smoke": frappe_pipeline_smoke(cfg),
        "sandwich_training": sandwich_training_smoke(cfg),
        "transcode_smoke": transcode_smoke(cfg),
        "encode_mpx_s_n12": sensor_encode_throughput_mpx_s(cfg, 12),
        "paper_tables": {
            "table1_matched_rate": table1_matched_rate_summary(),
            "deployment_tiers": table_deployment_tiers(),
            "standalone_kodak": table_standalone_kodak(),
        },
    }
