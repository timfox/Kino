"""OB3D — Omnidirectional Blender 3D benchmark (arXiv:2505.20126)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PAPER_ARXIV = "2505.20126"
PAPER_TITLE = "OB3D: A New Dataset for Benchmarking Omnidirectional 3D Reconstruction Using Blender"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
KAGGLE_URL = "https://www.kaggle.com/datasets/shintacs/ob3d-dataset"
GITHUB_URL = "https://github.com/gsisaoki/Omnidirectional_Blender_3D_Dataset"

ERP_WIDTH = 1600
ERP_HEIGHT = 800
IMAGES_PER_TRAJECTORY = 100
TRAIN_COUNT = 25
EVAL_COUNT = 25
NUM_SCENES = 12

SceneEnv = Literal["indoor", "outdoor"]
Trajectory = Literal["egocentric", "non_egocentric"]

OB3D_SCENES: dict[str, SceneEnv] = {
    "archiviz-flat": "indoor",
    "barbershop": "indoor",
    "bistro": "outdoor",
    "classroom": "indoor",
    "emerald-square": "outdoor",
    "fisher-hut": "outdoor",
    "lone-monk": "outdoor",
    "pavillion": "indoor",
    "restroom": "indoor",
    "san-miguel": "outdoor",
    "sponza": "outdoor",
    "sun-temple": "outdoor",
}

# Sparse point counts (Table 2) — ego / non-ego
SPARSE_POINTS: dict[str, tuple[int, int]] = {
    "archiviz-flat": (13_216, 9_537),
    "barbershop": (43_164, 40_635),
    "bistro": (48_554, 79_242),
    "classroom": (21_802, 23_343),
    "emerald-square": (34_835, 23_982),
    "fisher-hut": (6_087, 3_859),
    "lone-monk": (55_733, 61_013),
    "pavillion": (31_686, 43_855),
    "restroom": (23_754, 31_992),
    "san-miguel": (52_582, 47_088),
    "sponza": (53_340, 65_539),
    "sun-temple": (28_322, 37_932),
}


def train_indices() -> list[int]:
    return list(range(0, 100, 4))


def eval_indices() -> list[int]:
    return list(range(2, 100, 4))


@dataclass
class OB3DConfig:
    width: int = ERP_WIDTH
    height: int = ERP_HEIGHT
    max_depth: float = 50.0
    sky_depth_threshold: float = 45.0
