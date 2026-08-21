"""City-Mesh3R — simulation-ready city-scale mesh reconstruction (arXiv:2605.30310)."""

from ltx_trainer.city_mesh3r.config import CityMesh3RConfig
from ltx_trainer.city_mesh3r.layout import LIMITATIONS
from ltx_trainer.city_mesh3r.mesh_refine import mesh_objective, silhouette_loss
from ltx_trainer.city_mesh3r.paper_tables import (
    table_i_gauu_scene,
    table_ii_sfm_efficiency,
    table_iii_clustering_ablation,
    table_iv_poisson_ablation,
    table_v_conremesh_ablation,
    table_vi_garden_mesh_quality,
)
from ltx_trainer.city_mesh3r.pipeline import benchmarks_bundle, evaluation_demo, framework_card, pipeline_demo
from ltx_trainer.city_mesh3r.similarity import build_similarity_graph

__all__ = [
    "LIMITATIONS",
    "CityMesh3RConfig",
    "benchmarks_bundle",
    "build_similarity_graph",
    "evaluation_demo",
    "framework_card",
    "mesh_objective",
    "pipeline_demo",
    "silhouette_loss",
    "table_i_gauu_scene",
    "table_ii_sfm_efficiency",
    "table_iii_clustering_ablation",
    "table_iv_poisson_ablation",
    "table_v_conremesh_ablation",
    "table_vi_garden_mesh_quality",
]
