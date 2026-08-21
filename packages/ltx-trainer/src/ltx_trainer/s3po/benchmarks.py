"""Paper tables (Baniya et al., arXiv:2506.14803)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.s3po.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table I — 360VDS test (BD degradation, Y channel)
TABLE1_360VDS_BD = {
    "Bicubic": {"psnr": 25.22, "ssim": 0.7436, "ws_psnr": 24.46, "ws_ssim": 0.7177},
    "DUF": {"psnr": 26.46, "ssim": 0.817, "ws_psnr": 25.73, "ws_ssim": 0.7909},
    "RBPN": {"psnr": 27.14, "ssim": 0.8214, "ws_psnr": 26.27, "ws_ssim": 0.8004},
    "EDVR": {"psnr": 26.85, "ssim": 0.8147, "ws_psnr": 25.99, "ws_ssim": 0.7946},
    "BasicVSR": {"psnr": 27.05, "ssim": 0.8227, "ws_psnr": 26.19, "ws_ssim": 0.8027},
    "TGA": {"psnr": 27.31, "ssim": 0.8298, "ws_psnr": 26.39, "ws_ssim": 0.8095},
    "RSDN": {"psnr": 27.32, "ssim": 0.8313, "ws_psnr": 26.37, "ws_ssim": 0.8108},
    "RRN": {"psnr": 26.96, "ssim": 0.8214, "ws_psnr": 26.07, "ws_ssim": 0.8003},
    "S3PO": {"psnr": 27.51, "ssim": 0.8326, "ws_psnr": 26.52, "ws_ssim": 0.8120},
}

TABLE1_360VDS_BI = {
    "S3PO": {"psnr": 27.26, "ssim": 0.8227, "ws_psnr": 26.32, "ws_ssim": 0.8030},
    "RBPN": {"psnr": 27.14, "ssim": 0.8214, "ws_psnr": 26.27, "ws_ssim": 0.8004},
}

# Table II — MiG Panorama (4 clips, BI)
TABLE2_MIG = {
    "Bicubic": {"psnr": 29.00, "ssim": 0.8121, "ws_psnr": 28.81, "ws_ssim": 0.7964},
    "SMFN": {"psnr": 30.56, "ssim": 0.8505, "ws_psnr": 30.13, "ws_ssim": 0.8381},
    "BasicVSR": {"psnr": 30.89, "ssim": 0.8559, "ws_psnr": 30.22, "ws_ssim": 0.8452},
    "S3PO": {"psnr": 31.16, "ssim": 0.8565, "ws_psnr": 30.42, "ws_ssim": 0.8453},
}

# Table V — 360°-specific ablation (BD)
TABLE5_360_SPECIFIC = {
    "w/o_360_feat": {"psnr": 27.16, "ssim": 0.8225, "ws_psnr": 26.22, "ws_ssim": 0.8014},
    "w/o_weighted_loss": {"psnr": 27.46, "ssim": 0.8317, "ws_psnr": 26.46, "ws_ssim": 0.8105},
    "w/o_attention": {"psnr": 27.49, "ssim": 0.8322, "ws_psnr": 26.48, "ws_ssim": 0.8111},
    "S3PO_full": {"psnr": 27.51, "ssim": 0.8326, "ws_psnr": 26.52, "ws_ssim": 0.8120},
}

# Table VII — propagation + domain adaptation (BD)
TABLE7_PROPAGATION = {
    "w/o_hidden_state": {"psnr": 27.30, "ssim": 0.8266, "ws_psnr": 26.32, "ws_ssim": 0.8054},
    "w/o_mutual_exchange": {"psnr": 27.37, "ssim": 0.8287, "ws_psnr": 26.40, "ws_ssim": 0.8080},
    "w/o_domain_adaptation": {"psnr": 27.46, "ssim": 0.8317, "ws_psnr": 26.46, "ws_ssim": 0.8105},
    "S3PO_full": {"psnr": 27.51, "ssim": 0.8326, "ws_psnr": 26.52, "ws_ssim": 0.8120},
}

# Table VI — WSS-L1 β
TABLE6_BETA = {
    "beta_0.5": {"psnr": 27.50, "ws_ssim": 0.8118},
    "beta_1.0": {"psnr": 27.51, "ws_ssim": 0.8120},
    "beta_2.0": {"psnr": 27.49, "ws_ssim": 0.8116},
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1_360vds_bd": TABLE1_360VDS_BD,
        "table1_360vds_bi": TABLE1_360VDS_BI,
        "table2_mig": TABLE2_MIG,
        "table5_360_specific": TABLE5_360_SPECIFIC,
        "table7_propagation": TABLE7_PROPAGATION,
        "table6_beta": TABLE6_BETA,
    }
