"""Paper benchmark anchors (Tables 1–4, 6; arXiv:2605.25569)."""

from __future__ import annotations

from typing import Any

# Table 1 — LOL-v1 / LWSR (NIQE↓, CLIPIQA↑, MANIQA↑, MUSIQ↑)
TABLE1_LOL_V1: dict[str, dict[str, float]] = {
    "Retinexformer": {"NIQE": 3.455, "CLIPIQA": 0.429, "MANIQA": 0.383, "MUSIQ": 63.16},
    "CIDNet": {"NIQE": 4.110, "CLIPIQA": 0.488, "MANIQA": 0.511, "MUSIQ": 71.91},
    "LLFormer": {"NIQE": 3.580, "CLIPIQA": 0.331, "MANIQA": 0.317, "MUSIQ": 60.77},
    "DarkIR": {"NIQE": 5.335, "CLIPIQA": 0.389, "MANIQA": 0.410, "MUSIQ": 70.69},
    "QuadPrior": {"NIQE": 5.184, "CLIPIQA": 0.367, "MANIQA": 0.295, "MUSIQ": 58.81},
    "CLE Diffusion": {"NIQE": 4.893, "CLIPIQA": 0.581, "MANIQA": 0.435, "MUSIQ": 68.84},
    "ControlLight": {"NIQE": 4.567, "CLIPIQA": 0.553, "MANIQA": 0.512, "MUSIQ": 70.20},
}

TABLE1_LWSR: dict[str, dict[str, float]] = {
    "Retinexformer": {"NIQE": 3.778, "CLIPIQA": 0.420, "MANIQA": 0.401, "MUSIQ": 58.48},
    "CIDNet": {"NIQE": 3.708, "CLIPIQA": 0.415, "MANIQA": 0.387, "MUSIQ": 56.25},
    "LLFormer": {"NIQE": 3.791, "CLIPIQA": 0.394, "MANIQA": 0.360, "MUSIQ": 57.44},
    "DarkIR": {"NIQE": 4.103, "CLIPIQA": 0.462, "MANIQA": 0.431, "MUSIQ": 64.45},
    "QuadPrior": {"NIQE": 5.045, "CLIPIQA": 0.345, "MANIQA": 0.358, "MUSIQ": 58.91},
    "CLE Diffusion": {"NIQE": 4.265, "CLIPIQA": 0.491, "MANIQA": 0.388, "MUSIQ": 62.63},
    "ControlLight": {"NIQE": 4.232, "CLIPIQA": 0.589, "MANIQA": 0.494, "MUSIQ": 68.39},
}

# Table 2 — DICM / LIME / RealIR-Bench
TABLE2_DICM: dict[str, dict[str, float]] = {
    "Retinexformer": {"NIQE": 3.962, "CLIPIQA": 0.377, "MANIQA": 0.291, "MUSIQ": 54.27},
    "CIDNet": {"NIQE": 3.657, "CLIPIQA": 0.501, "MANIQA": 0.384, "MUSIQ": 57.90},
    "LLFormer": {"NIQE": 3.943, "CLIPIQA": 0.435, "MANIQA": 0.274, "MUSIQ": 55.03},
    "DarkIR": {"NIQE": 3.869, "CLIPIQA": 0.463, "MANIQA": 0.345, "MUSIQ": 57.44},
    "QuadPrior": {"NIQE": 4.797, "CLIPIQA": 0.488, "MANIQA": 0.315, "MUSIQ": 58.21},
    "CLE Diffusion": {"NIQE": 4.368, "CLIPIQA": 0.390, "MANIQA": 0.218, "MUSIQ": 47.38},
    "ControlLight": {"NIQE": 3.522, "CLIPIQA": 0.698, "MANIQA": 0.505, "MUSIQ": 68.22},
}

TABLE2_LIME: dict[str, dict[str, float]] = {
    "ControlLight": {"NIQE": 3.638, "CLIPIQA": 0.576, "MANIQA": 0.526, "MUSIQ": 67.68},
}

TABLE2_REALIR: dict[str, dict[str, float]] = {
    "ControlLight": {"NIQE": 3.748, "CLIPIQA": 0.550, "MANIQA": 0.491, "MUSIQ": 67.96},
}

# Table 3 — linear control (δ_smooth↓, CLIP-Dir↑)
TABLE3_LINEAR: dict[str, dict[str, dict[str, float]]] = {
    "RealIR-Bench": {
        "ConceptSlider": {"delta_smooth": 0.9237, "CLIP-Dir": -0.0530},
        "AttributeControl": {"delta_smooth": 0.7262, "CLIP-Dir": 0.3520},
        "KSlider": {"delta_smooth": 0.1956, "CLIP-Dir": 0.0901},
        "SliderEdit": {"delta_smooth": 0.3840, "CLIP-Dir": -0.3125},
        "CLE Diffusion": {"delta_smooth": 0.7503, "CLIP-Dir": -0.2624},
        "ControlLight": {"delta_smooth": 0.2195, "CLIP-Dir": 0.9138},
    },
    "DICM": {
        "ControlLight": {"delta_smooth": 0.2382, "CLIP-Dir": 0.9012},
    },
    "LIME": {
        "ControlLight": {"delta_smooth": 0.1786, "CLIP-Dir": 0.9159},
    },
}

# Table 4 — LwFM ablation on RealIR low-light subset
TABLE4_LWFM_ABLATION: dict[str, float] = {
    "L_FM_LI-LPIPS": 0.2237,
    "L_FM_NIQE": 5.6242,
    "L_FM_MANIQA": 0.3384,
    "L_FM_MUSIQ": 55.2252,
    "L_FM_CLIPIQA": 0.5232,
    "LwFM_LI-LPIPS": 0.2148,
    "LwFM_NIQE": 4.5367,
    "LwFM_MANIQA": 0.4180,
    "LwFM_MUSIQ": 62.5262,
    "LwFM_CLIPIQA": 0.6112,
}

# Table 6 — interpolation ablation (NIQE↓ / MUSIQ↑ at I0, I0.2 … I1)
TABLE6_INTERPOLATION: dict[str, dict[str, list[float]]] = {
    "NIQE": {
        "Alpha Blending": [4.588, 3.931, 3.561, 3.356, 3.461, 3.695],
        "Retinex (Ours)": [4.588, 4.171, 3.649, 3.315, 3.419, 3.695],
    },
    "MUSIQ": {
        "Alpha Blending": [55.936, 62.620, 66.289, 70.047, 68.469, 70.019],
        "Retinex (Ours)": [55.936, 58.780, 60.889, 67.626, 67.716, 70.019],
    },
}

STRENGTH_GRID_TABLE6: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": {"LOL-v1": TABLE1_LOL_V1, "LWSR": TABLE1_LWSR},
        "table2": {"DICM": TABLE2_DICM, "LIME": TABLE2_LIME, "RealIR-Bench": TABLE2_REALIR},
        "table3": TABLE3_LINEAR,
        "table4_lwfm": TABLE4_LWFM_ABLATION,
        "table6_interpolation": TABLE6_INTERPOLATION,
        "eval_strengths": [0.25, 0.5, 0.75, 1.0],
        "training_strengths": [0.2, 0.4, 0.6, 0.8, 1.0],
        "website": "https://yfyang007.github.io/ControlLight/",
        "arxiv": "arXiv:2605.25569",
    }
