"""Paper tables (Wang et al., arXiv:2505.16862, NeurIPS 2025)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.par.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — text-to-panorama on Matterport3D
TABLE1_T2P = {
    "PanFusion": {"params": "1.9B", "T2P": True, "PO": False, "PE": False, "FAED": 5.12, "FID": 45.21, "CLIP": 31.07, "DS": 0.71},
    "Text2Light": {"params": "0.8B", "T2P": True, "PO": False, "PE": False, "FAED": 68.90, "FID": 70.42, "CLIP": 27.90, "DS": 7.55},
    "PanoLlama": {"params": "0.8B", "T2P": True, "PO": False, "PE": False, "FAED": 33.15, "FID": 103.51, "CLIP": 32.54, "DS": 13.99},
    "PAR-0.3B": {"params": "0.3B", "T2P": True, "PO": True, "PE": True, "FAED": 3.39, "FID": 41.15, "CLIP": 30.21, "DS": 0.58},
    "PAR-0.6B": {"params": "0.6B", "T2P": True, "PO": True, "PE": True, "FAED": 3.34, "FID": 39.31, "CLIP": 30.34, "DS": 0.57},
    "PAR-1.4B": {"params": "1.4B", "T2P": True, "PO": True, "PE": True, "FAED": 3.75, "FID": 37.37, "CLIP": 30.41, "DS": 0.58},
}

# Table 2 — panorama outpainting on Matterport3D
TABLE2_PO = {
    "AOG-Net": {"T2P": False, "PO": True, "PE": False, "FID": 83.02, "FID_h": 37.88},
    "2S-ODIS": {"T2P": False, "PO": True, "PE": False, "FID": 52.59, "FID_h": 35.18},
    "PAR w/o prompt": {"T2P": True, "PO": True, "PE": True, "FID": 41.63, "FID_h": 25.97},
    "PAR w/ prompt": {"T2P": True, "PO": True, "PE": True, "FID": 32.68, "FID_h": 12.20},
}

# Table 3 — cyclic consistency ablation (PAR-1.4B T2P)
TABLE3_CONSISTENCY = {
    "w/o L_consistency": {"FID": 39.55, "CLIP": 30.25, "DS": 0.57},
    "w/ L_consistency": {"FID": 37.37, "CLIP": 30.41, "DS": 0.58},
}

# Table 4 — inference time vs AR steps (0.3B)
TABLE4_AR_STEPS = {
    16: 3.02,
    32: 5.49,
    64: 10.03,
}

MODEL_SIZES = ("0.3B", "0.6B", "1.4B")


def par_beats_ar_baseline(metric: str = "FID") -> bool:
    ours = TABLE1_T2P["PAR-0.3B"][metric]
    return ours < TABLE1_T2P["PanoLlama"][metric] and ours < TABLE1_T2P["Text2Light"][metric]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1_t2p": TABLE1_T2P,
        "table2_po": TABLE2_PO,
        "table3_consistency": TABLE3_CONSISTENCY,
        "table4_ar_steps": TABLE4_AR_STEPS,
    }
