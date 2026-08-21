"""City-Mesh3R framework card, pipeline demo, benchmarks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.city_mesh3r.clustering import (
    cluster_overlap_matrix,
    minimum_spanning_tree_edges,
    slpa_overlapping_communities,
)
from ltx_trainer.city_mesh3r.config import CityMesh3RConfig
from ltx_trainer.city_mesh3r.layout import LIMITATIONS
from ltx_trainer.city_mesh3r.mesh_refine import mesh_objective, normal_rotation_rate, projected_target_edge_length
from ltx_trainer.city_mesh3r.metrics import mesh_quality_bundle, precision_recall_f1
from ltx_trainer.city_mesh3r.partition import fit_dominant_plane_ransac, grid_partitions, project_to_plane, rank_cameras_for_partition
from ltx_trainer.city_mesh3r.paper_tables import (
    table_i_gauu_scene,
    table_ii_sfm_efficiency,
    table_iii_clustering_ablation,
    table_iv_poisson_ablation,
    table_v_conremesh_ablation,
    table_vi_garden_mesh_quality,
)
from ltx_trainer.city_mesh3r.similarity import build_similarity_graph, dinov2_feature_smoke
from ltx_trainer.city_mesh3r.stitching import clip_mesh_to_exterior, seam_vertices_in_overlap, stitch_meshes_vertex_count


def framework_card(cfg: CityMesh3RConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CityMesh3RConfig()
    return {
        "name": "City-Mesh3R",
        "paper": cfg.paper_arxiv,
        "lab": cfg.lab,
        "idea": (
            "End-to-end images-to-mesh pipeline for city-scale scenes: DINOv2 view graph, "
            "SLPA overlapping clusters + distributed COLMAP/MASt3R sparse SfM, support-plane "
            "spatial partitioning with geometry-aware camera selection, TSDF/Poisson init, "
            "curvature-aware adaptive remeshing, and partition stitching."
        ),
        "stages": [
            "Images → Sparse SfM (global features, SLPA, cluster-wise MASt3R+COLMAP, Sim(3) merge)",
            "Sparse SfM → Mesh (grid partition, camera ranking, dense MASt3R + alignment, Poisson)",
            "Mesh refinement (silhouette + monocular normals, curvature-guided remeshing)",
            "Mesh stitching (exterior clip + seam Delaunay + weld)",
        ],
        "datasets": list(cfg.datasets),
        "baselines": list(cfg.baselines),
        "limitations": list(LIMITATIONS),
    }


def pipeline_demo(cfg: CityMesh3RConfig | None = None, *, seed: int = 7) -> dict[str, Any]:
    cfg = cfg or CityMesh3RConfig()
    torch.manual_seed(seed)
    n_img = 48
    feats = dinov2_feature_smoke(n_img, dim=32, seed=seed)
    adj = build_similarity_graph(feats, threshold=cfg.similarity_threshold)
    clusters = slpa_overlapping_communities(adj)
    overlap = cluster_overlap_matrix(clusters, n_img)
    mst = minimum_spanning_tree_edges(overlap)

    points = torch.randn(500, 3) * 10
    points[:, 2] = points[:, 2].abs() * 0.1
    n_plane, d_plane = fit_dominant_plane_ransac(points)
    origin = -d_plane * n_plane
    u = torch.tensor([1.0, 0.0, 0.0])
    v = torch.linalg.cross(n_plane, u)
    v = v / v.norm()
    u = torch.linalg.cross(v, n_plane)
    uv = project_to_plane(points, origin, u, v)
    masks = grid_partitions(
        uv,
        rows=cfg.partition_grid_rows,
        cols=cfg.partition_grid_cols,
        inflate_u=cfg.partition_inflation_u,
        inflate_v=cfg.partition_inflation_v,
    )
    obs = {i: [i % 8, (i + 1) % 8] for i in range(500)}
    top_cams = rank_cameras_for_partition(
        torch.arange(100),
        obs,
        top_m=cfg.top_cameras_per_partition,
    )

    h, w = 32, 32
    tgt_sil = torch.zeros(h, w)
    tgt_sil[8:24, 8:24] = 1.0
    pred_sil = tgt_sil + 0.05 * torch.randn(h, w)
    tgt_n = torch.zeros(h, w, 3)
    tgt_n[..., 2] = 1.0
    pred_n = tgt_n + 0.02 * torch.randn(h, w, 3)
    losses = mesh_objective(
        pred_sil,
        tgt_sil,
        pred_n,
        tgt_n,
        lambda_n=cfg.lambda_normal,
        lambda_s=cfg.lambda_silhouette,
    )
    rot = normal_rotation_rate(tgt_n)
    p_tgt = projected_target_edge_length(rot, theta0=cfg.normal_rotation_tol_rad)

    v1 = torch.randn(20, 3)
    v2 = torch.randn(18, 3)
    f1 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    b1_min, b1_max = v1.min(0).values - 0.5, v1.max(0).values + 0.5
    b2_min, b2_max = v2.min(0).values - 0.5, v2.max(0).values + 0.5
    seam = seam_vertices_in_overlap(v1, b1_min, b1_max, b2_min, b2_max)
    stitch_n = stitch_meshes_vertex_count(v1, v2, seam)
    ext_v, ext_f = clip_mesh_to_exterior(v1, f1, b2_min, b2_max)

    pr = precision_recall_f1(tp=90.0, fp=10.0, fn=8.0)
    mq = mesh_quality_bundle(ext_v, ext_f)

    return {
        "num_images": n_img,
        "num_clusters": len(clusters),
        "mst_edges": len(mst),
        "num_partitions": len(masks),
        "top_cameras_sample": top_cams[:5],
        "mesh_losses": {k: round(float(v.detach()), 4) for k, v in losses.items()},
        "mean_target_edge_px": round(float(p_tgt.mean()), 4),
        "stitch_vertex_count": stitch_n,
        "mesh_quality": mq,
        "prf1_smoke": {k: round(v, 4) for k, v in pr.items()},
    }


def evaluation_demo(cfg: CityMesh3RConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    t1 = table_i_gauu_scene()
    ours_lower = t1[0]["Ours"]
    demo["framework"] = framework_card(cfg)
    demo["paper_f1_cuhk_lower"] = ours_lower["F1"]
    demo["paper_sfm_hours"] = table_ii_sfm_efficiency()[2]["time_hrs"]
    demo["limitations"] = LIMITATIONS
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    t1 = table_i_gauu_scene()
    t2 = table_ii_sfm_efficiency()
    return {
        "table_i_gauu_scene": t1,
        "table_ii_sfm": t2,
        "table_iii_clustering": table_iii_clustering_ablation(),
        "table_iv_poisson": table_iv_poisson_ablation(),
        "table_v_conremesh": table_v_conremesh_ablation(),
        "table_vi_garden_quality": table_vi_garden_mesh_quality(),
        "headline": {
            "best_f1_scenes": ["CUHK-LOWER", "SZIIT"],
            "sfm_speedup_vs_colmap_hrs": t2[0]["time_hrs"] / t2[2]["time_hrs"],
            "sziit_f1_ours": t1[3]["Ours"]["F1"],
            "garden_aspect_ratio_ours": table_vi_garden_mesh_quality()["Ours"]["AR"],
        },
    }
