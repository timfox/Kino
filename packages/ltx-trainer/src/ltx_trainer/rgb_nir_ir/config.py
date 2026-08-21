"""Ambient-robust RGB–NIR inverse rendering (SIGGRAPH 2026 / arXiv:2605.30250)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RgbNirIrConfig:
    paper_arxiv: str = "arXiv:2605.30250"
    venue: str = "SIGGRAPH Conference Papers 2026"
    doi: str = "10.1145/3799902.3811078"
    lab: str = "POSTECH"
    camera: str = "JAI FS-1600 (pixel-aligned RGB–NIR)"
    nir_flash: str = "Advanced Illumination AL295"
    robot: tuple[str, ...] = ("AgileX Piper arm", "AgileX Ranger Mini V2 base")
    nir_basis_count: int = 4
    stages: tuple[str, ...] = (
        "geometry_init_rgb_2dgs",
        "nir_flash_inverse_rendering",
        "rgb_environment_inverse_rendering",
    )
    baselines: tuple[str, ...] = (
        "NeILF",
        "TensoIR",
        "GS-IR",
        "R3DG",
        "IRGS",
        "WildLight",
        "MaterialFusion",
    )
    real_objects: int = 4
    real_environments: int = 4
    synthetic_objects: int = 4
    synthetic_environments: int = 4
    views_per_scene_min: int = 100
