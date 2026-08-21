"""City-Mesh3R: simulation-ready city-scale mesh reconstruction (arXiv:2605.30310)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CityMesh3RConfig:
    paper_arxiv: str = "arXiv:2605.30310"
    lab: str = "TCS Research Visual Computing & Embodied AI"
    global_descriptor: str = "dinov2"
    clustering: str = "slpa"
    sparse_matcher: str = "mast3r"
    sparse_mapper: str = "colmap"
    dense_depth: str = "mast3r"
    normal_estimator: str = "moge"
    similarity_threshold: float = 0.55
    partition_grid_rows: int = 4
    partition_grid_cols: int = 4
    partition_inflation_u: float = 0.1
    partition_inflation_v: float = 0.1
    top_cameras_per_partition: int = 32
    lambda_normal: float = 1.0
    lambda_silhouette: float = 1.0
    normal_rotation_tol_rad: float = 0.15
    remesh_velocity_ref: float = 0.02
    remesh_slack_gain: float = 0.5
    stitch_epsilon: float = 0.05
    datasets: tuple[str, ...] = (
        "GauU-Scene/CUHK-LOWER",
        "GauU-Scene/CUHK-UPPER",
        "GauU-Scene/LFLS",
        "GauU-Scene/SZIIT",
        "UrbanScene3D/Residence",
    )
    baselines: tuple[str, ...] = (
        "CityGaussianV2",
        "CityGS-X",
        "MASt3R-COLMAP",
        "MASt3R-GLOMAP",
        "ConRemesh",
    )
