"""TriSplat: simulation-ready feed-forward triangle mesh reconstruction (Wang et al. arXiv:2605.26115)."""

from ltx_trainer.trisplat.config import TriSplatConfig
from ltx_trainer.trisplat.export import export_mesh_obj, export_mesh_ply
from ltx_trainer.trisplat.inference import (
    TriSplatMeshExport,
    export_simulation_mesh,
    reconstruct_from_images,
    reconstruct_from_sfm,
    reconstruct_view_triangles,
    render_novel_view,
)
from ltx_trainer.trisplat.sfm_bridge import TriSplatSceneResult

__all__ = [
    "TriSplatConfig",
    "TriSplatMeshExport",
    "TriSplatSceneResult",
    "export_mesh_obj",
    "export_mesh_ply",
    "export_simulation_mesh",
    "reconstruct_from_images",
    "reconstruct_from_sfm",
    "reconstruct_view_triangles",
    "render_novel_view",
]
