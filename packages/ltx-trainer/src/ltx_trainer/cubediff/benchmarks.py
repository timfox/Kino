"""Paper Table 1 & 2 (Kalischek et al., ICLR 2025, arXiv:2501.17162)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cubediff.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL

# Table 1 — Laval Indoor (excerpt)
TABLE1_LAVAL = {
    "Text2Light": {"fid": 28.3, "kid_x100": 1.45, "clip_fid": 11.5, "faed": 136.1, "cs": 25.18},
    "PanFusion": {"fid": 41.7, "kid_x100": 2.85, "clip_fid": 19.8, "faed": 71.7, "cs": 26.58},
    "OmniDreamer": {"fid": 71.0, "kid_x100": 5.17, "clip_fid": 23.9, "faed": 19.2},
    "PanoDiffusion": {"fid": 58.6, "kid_x100": 4.08, "clip_fid": 26.6, "faed": 106.8},
    "Ours_img": {"fid": 11.7, "kid_x100": 0.47, "clip_fid": 4.4, "faed": 22.0},
    "Diffusion360": {"fid": 33.1, "kid_x100": 2.07, "clip_fid": 16.9, "faed": 23.7, "cs": 26.38},
    "Ours_img_txt": {"fid": 9.5, "kid_x100": 0.32, "clip_fid": 3.2, "faed": 18.4, "cs": 27.02},
    "MVDiffusion": {"fid": 25.7, "kid_x100": 1.11, "clip_fid": 13.5, "cs": 27.44},
    "Ours_img_multitxt": {"fid": 10.0, "kid_x100": 0.35, "clip_fid": 4.1, "faed": 21.2, "cs": 30.17},
}

TABLE1_SUN360 = {
    "Ours_img_txt": {"fid": 25.5, "kid_x100": 1.33, "clip_fid": 8.1, "faed": 7.6, "cs": 25.00},
    "Ours_img_multitxt": {"fid": 24.1, "kid_x100": 1.33, "clip_fid": 7.0, "faed": 5.7, "cs": 28.14},
    "MVDiffusion": {"fid": 50.9, "kid_x100": 3.71, "clip_fid": 15.4, "faed": 32.3, "cs": 25.54},
    "PanFusion": {"fid": 30.0, "kid_x100": 1.42, "clip_fid": 7.8, "faed": 44.5, "cs": 25.28},
}

# Table 2 appendix — data ablation Laval
TABLE2_DATA_ABLATION_LAVAL = {
    "Ourstiny": {"fid": 27.3, "kid_x100": 1.05, "clip_fid": 8.8},
    "Oursmedium": {"fid": 13.8, "kid_x100": 0.66, "clip_fid": 8.5},
    "Oursfull": {"fid": 10.0, "kid_x100": 0.35, "clip_fid": 4.1},
    "MVDiffusion": {"fid": 25.7, "kid_x100": 1.11, "clip_fid": 13.5},
}

from ltx_trainer.cubediff.config import TRAIN_ITERS, TRAIN_PANORAMAS  # noqa: F401


def cubediff_beats_mvdiffusion_laval() -> bool:
    return TABLE1_LAVAL["Ours_img_txt"]["fid"] < TABLE1_LAVAL["MVDiffusion"]["fid"]


def cubediff_best_fid_laval() -> bool:
    ours = TABLE1_LAVAL["Ours_img_txt"]["fid"]
    return all(ours <= row["fid"] for k, row in TABLE1_LAVAL.items() if k.startswith("Ours") or k in ("MVDiffusion", "Diffusion360", "Text2Light"))


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "table1_laval": TABLE1_LAVAL,
        "table1_sun360": TABLE1_SUN360,
        "table2_data_ablation": TABLE2_DATA_ABLATION_LAVAL,
        "train_panoramas": TRAIN_PANORAMAS,
    }
