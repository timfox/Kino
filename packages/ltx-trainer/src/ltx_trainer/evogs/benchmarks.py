"""Paper table anchors (Tables 1–5, Fig. 7)."""

from __future__ import annotations

from typing import Any

LevelMetrics = dict[str, float | int]


def _row(psnr: float, ssim: float, lpips: float, stor: float, mem: float) -> LevelMetrics:
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips, "stor_mb": stor, "mem_mb": mem}


TABLE1: dict[str, dict[str, dict[str, LevelMetrics]]] = {
    "Blender": {
        "Monolithic": {
            "L0": _row(36.19, 0.982, 0.020, 18.96, 18.96),
            "L3": _row(33.72, 0.970, 0.027, 147.13, 147.13),
        },
        "LapisGS": {
            "L0": _row(36.18, 0.982, 0.020, 18.98, 18.98),
            "L3": _row(32.52, 0.962, 0.045, 133.90, 133.90),
        },
        "EvoGS": {
            "L0": _row(36.29, 0.983, 0.020, 21.25, 15.14),
            "L3": _row(32.71, 0.964, 0.047, 62.59, 23.97),
        },
    },
    "Mip-NeRF360": {
        "LapisGS": {
            "L0": _row(28.73, 0.888, 0.105, 240.52, 240.52),
            "L3": _row(26.68, 0.762, 0.268, 1382.05, 1382.05),
        },
        "EvoGS": {
            "L0": _row(29.04, 0.889, 0.100, 157.83, 123.20),
            "L3": _row(26.65, 0.753, 0.276, 606.73, 385.25),
        },
    },
    "Tanks&Temples": {
        "LapisGS": {"L3": _row(23.42, 0.818, 0.228, 691.44, 691.44)},
        "EvoGS": {"L3": _row(23.42, 0.808, 0.237, 354.84, 149.70)},
    },
    "Deep Blending": {
        "LapisGS": {"L3": _row(28.97, 0.891, 0.276, 999.82, 999.82)},
        "EvoGS": {"L3": _row(29.10, 0.887, 0.277, 405.30, 215.43)},
    },
}


def table1_dataset(dataset: str) -> dict[str, Any]:
    return TABLE1.get(dataset, {})


def table1_evogs_beats_lapis_storage(dataset: str = "Blender", level: str = "L3") -> bool:
    d = TABLE1.get(dataset, {})
    if "EvoGS" not in d or "LapisGS" not in d:
        return True
    if level not in d["EvoGS"] or level not in d["LapisGS"]:
        return True
    return float(d["EvoGS"][level]["stor_mb"]) < float(d["LapisGS"][level]["stor_mb"])


def table2_sym_vs_asym() -> dict[str, dict[str, float]]:
    """Table 2 average across datasets."""
    return {
        "Sym.": {"L0": 28.84, "L1": 28.64, "L2": 27.87, "L3": 26.72},
        "Asym.": {"L0": 29.66, "L1": 29.55, "L2": 28.86, "L3": 27.73},
        "stor_sym_L3": 344.49,
        "stor_asym_L3": 347.36,
    }


def table3_ghost_splat_ratios() -> dict[str, dict[str, float]]:
    """Table 3 transparent splat ratios (%)."""
    return {
        "LapisGS": {"LOD0": 15.92, "LOD1": 46.91, "LOD2": 55.59, "LOD3": 65.83},
        "EvoGS (Sym.)": {"LOD0": 14.40, "LOD1": 25.99, "LOD2": 33.67, "LOD3": 38.85},
        "EvoGS": {"LOD0": 12.08, "LOD1": 17.02, "LOD2": 21.04, "LOD3": 24.92},
    }


def table4_lapis_layer_breakdown() -> dict[str, float]:
    """Table 4 LapisGS LOD3 layer transparency (%)."""
    return {"layer0": 93.82, "layer1": 88.66, "layer2": 72.55, "layer3": 14.20, "total": 65.83}


def table5_compressibility() -> dict[str, dict[str, float]]:
    """Table 5 average storage (MB) at L3."""
    return {
        "LapisGS": {"psnr": 27.64, "stor_mb": 801.80},
        "EvoGS": {"psnr": 27.73, "stor_mb": 347.36},
        "EvoGS w/ Comp.": {"psnr": 27.63, "stor_mb": 91.75},
    }


def playroom_qualitative_ssim() -> dict[str, dict[str, float]]:
    """Figure 8 Playroom SSIM at L0–L3."""
    return {
        "LapisGS": {"L0": 0.64, "L1": 0.71, "L2": 0.80, "L3": 0.86},
        "EvoGS (Asym.)": {"L0": 0.89, "L1": 0.91, "L2": 0.91, "L3": 0.91},
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "table2_sym_asym": table2_sym_vs_asym(),
        "table3_ghost": table3_ghost_splat_ratios(),
        "table4_lapis_layers": table4_lapis_layer_breakdown(),
        "table5_compression": table5_compressibility(),
        "fig8_playroom": playroom_qualitative_ssim(),
    }


def ours_beats_baselines() -> dict[str, bool]:
    t3 = table3_ghost_splat_ratios()
    t5 = table5_compressibility()
    t2 = table2_sym_vs_asym()
    return {
        "ghost_under_25pct": t3["EvoGS"]["LOD3"] < 25.0,
        "ghost_below_lapis": t3["EvoGS"]["LOD3"] < t3["LapisGS"]["LOD3"],
        "asym_beats_sym_L3": t2["Asym."]["L3"] > t2["Sym."]["L3"],
        "compressed_under_100mb": t5["EvoGS w/ Comp."]["stor_mb"] < 100.0,
        "storage_beats_lapis": t5["EvoGS"]["stor_mb"] < t5["LapisGS"]["stor_mb"],
        "blender_L3_storage": table1_evogs_beats_lapis_storage("Blender", "L3"),
    }
