"""PhysHDR-GS hyperparameters (Sec. 4.4, 5.1, D)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExposureSetting(str, Enum):
    EXP3 = "exp3"  # random t in {t1,t3,t5} each iteration
    EXP1 = "exp1"  # fixed random t from {t1,t3,t5}


@dataclass(frozen=True)
class PhysHDRConfig:
    num_gaussians: int = 256
    image_size: int = 64
    hidden_dim: int = 32
    init_radius: float = 1.5
    max_gaussians: int = 20_000
    max_gaussians_render: int = 1024
    use_perspective: bool = True
    fast_raster: bool = True
    num_train_views: int = 8
    image_scale: float = 1.0
    lambda_rec: float = 1.0
    lambda_cons: float = 0.5
    lambda_unit: float = 0.0  # 0.5 for synthetic
    gamma_mse: float = 0.2
    scale_s: float = 1.0  # illumination-guided gradient scaling strength
    densify_tau: float = 2e-4
    densify_interval: int = 100
    densify_from_iter: int = 200
    max_iterations: int = 30_000
    freeze_fmix_iters: int = 10_000
    lr_gaussians: float = 1e-3
    lr_radiance: float = 6e-5
    lr_tonemap: float = 2e-4
    log_interval: int = 100
    use_gi_branch: bool = True
    use_hdr_cons: bool = True
    use_igs: bool = True
    exposure_setting: str = "exp3"
    backbone: str = "3dgs"  # or "scaffold_gs"
    scene_dir: str | None = None


# Five exposure times (relative); LDR-OE = {t1,t3,t5}, LDR-NE = {t2,t4}
EXPOSURE_TIMES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
LDR_OE_INDICES = (0, 2, 4)
LDR_NE_INDICES = (1, 3)

PAPER_URL = "https://arxiv.org/abs/2603.28020"
PROJECT_URL = "https://huimin-zeng.github.io/PhysHDR-GS/"
