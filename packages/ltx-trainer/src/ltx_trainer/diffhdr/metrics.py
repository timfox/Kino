"""Paper benchmark metrics (Tables 1–5)."""

from __future__ import annotations

TABLE1_SI_HDR: dict[str, dict[str, float]] = {
    "ours": {"hdr_vdp3": 6.98, "pu21_piqe": 19.37, "fid": 18.68},
    "lediff": {"hdr_vdp3": 6.56, "pu21_piqe": 22.71, "fid": 25.98},
    "singlehdr": {"hdr_vdp3": 7.37, "pu21_piqe": 26.64, "fid": 27.55},
    "hdrcnn": {"hdr_vdp3": 6.82, "pu21_piqe": 24.30, "fid": 19.26},
}

TABLE2_VIDEO: dict[str, dict[str, float]] = {
    "ours_cinematic": {"fovvideovdp": 6.89, "dover": 0.81, "musiq": 58.38, "clipiqa": 0.41},
    "ours_polyhaven": {"fovvideovdp": 7.65, "dover": 0.68, "musiq": 60.02, "clipiqa": 0.50},
    "lediff_cinematic": {"fovvideovdp": 3.75, "dover": 0.63, "musiq": 47.90, "clipiqa": 0.28},
}

TABLE3_IN_THE_WILD: dict[str, dict[str, float]] = {
    "ours": {"dover": 0.74, "musiq": 55.79, "clipiqa": 0.48},
    "singlehdr": {"dover": 0.71, "musiq": 53.21, "clipiqa": 0.46},
    "lediff": {"dover": 0.61, "musiq": 53.68, "clipiqa": 0.42},
}

TABLE4_LOG_GAMMA: dict[str, dict[str, float]] = {
    "linear": {"psnr": 22.16, "ssim": 0.74, "lpips": 0.28},
    "log": {"psnr": 14.61, "ssim": 0.74, "lpips": 0.57},
    "ours_wo_gamma": {"psnr": 25.38, "ssim": 0.75, "lpips": 0.34},
    "ours": {"psnr": 32.86, "ssim": 0.86, "lpips": 0.15},
}

TABLE5_ABLATION: dict[str, dict[str, float]] = {
    "ours_wo_data_aug": {"fovvideovdp": 7.57, "dover": 0.67, "musiq": 59.86, "clipiqa": 0.49},
    "ours_wo_mask": {"fovvideovdp": 7.58, "dover": 0.67, "musiq": 59.42, "clipiqa": 0.48},
    "ours": {"fovvideovdp": 7.65, "dover": 0.68, "musiq": 60.02, "clipiqa": 0.50},
}
