"""Training / retrieval demos."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.hypevpr.benchmarks import TABLE2_PITTS, TABLE4_MANIFOLD
from ltx_trainer.hypevpr.config import HypeVPRConfig
from ltx_trainer.hypevpr.hierarchy import num_windows_at_level
from ltx_trainer.hypevpr.hypevpr_net import HypeVPRStub
from ltx_trainer.hypevpr.losses import total_loss
from ltx_trainer.hypevpr.poincare import poincare_distance
from ltx_trainer.hypevpr.retrieval import coarse_retrieve, hierarchical_rerank_score
from ltx_trainer.hypevpr.synthetic import synthetic_panorama, synthetic_query


def evaluation_demo_run(cfg: HypeVPRConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HypeVPRConfig()
    model = HypeVPRStub(cfg)
    q = synthetic_query(cfg)
    p = synthetic_panorama(cfg)
    with torch.no_grad():
        out = model(q, p)
    tree = out["db_tree"]
    assert isinstance(tree, dict)
    return {
        "query_shape": list(q.shape),
        "pano_shape": list(p.shape),
        "hq_norm": float(out["hq"].norm().item()),
        "num_windows": num_windows_at_level(cfg.hierarchy_levels),
        "levels_in_tree": sorted(tree.keys()),
        "top_descriptor_dim": int(out["h_top"].shape[-1]),
    }


def train_step(cfg: HypeVPRConfig | None = None) -> dict[str, float]:
    cfg = cfg or HypeVPRConfig()
    model = HypeVPRStub(cfg)
    q = synthetic_query(cfg)
    p_pos = synthetic_panorama(cfg)
    p_neg = torch.roll(p_pos, shifts=cfg.query_size, dims=-1)
    out_q = model(q, p_pos)
    out_neg = model.encode_database(p_neg)
    hq = out_q["hq"]
    assert isinstance(out_q["db_tree"], dict)
    h_pos = out_q["h_top"]
    h_neg = out_neg[1][0]
    losses = total_loss(
        hq.squeeze(0),
        out_q["db_tree"],
        h_pos,
        h_neg,
        margin=cfg.triplet_margin,
        c=cfg.curvature,
    )
    return {k: float(v.detach()) for k, v in losses.items()}


def retrieval_demo(cfg: HypeVPRConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HypeVPRConfig()
    model = HypeVPRStub(cfg)
    q = synthetic_query(cfg)
    with torch.no_grad():
        _, hq = model.encode_query(q)
        hq = hq.squeeze(0)
        db_trees = []
        tops = []
        for i in range(12):
            pano = synthetic_panorama(cfg)
            if i > 0:
                pano = torch.roll(pano, shifts=i * 7, dims=-1)
            tree = model.encode_database(pano)
            db_trees.append(tree)
            tops.append(model.top_descriptor(tree))
        db_tops = torch.stack(tops, dim=0)
        idx, _ = coarse_retrieve(hq, db_tops, c=cfg.curvature, top_k_prime=cfg.top_k_prime)
        cand = idx.view(-1).tolist()
        scores = [
            hierarchical_rerank_score(
                hq,
                db_trees[i],
                levels=cfg.retrieval_levels,
                weights=cfg.level_weights,
                c=cfg.curvature,
            )
            for i in cand
        ]
    return {
        "coarse_top3": idx[:3].tolist(),
        "rerank_scores_head": scores[:3],
        "hq_db0_distance": float(poincare_distance(hq.unsqueeze(0), db_tops[0:1], c=cfg.curvature).item()),
    }


def norm_hierarchy_demo(cfg: HypeVPRConfig | None = None) -> dict[str, float]:
    """Higher level → smaller norm on Poincaré ball (Fig. 4)."""
    cfg = cfg or HypeVPRConfig()
    model = HypeVPRStub(cfg)
    p = synthetic_panorama(cfg)
    with torch.no_grad():
        tree = model.encode_database(p)
    norms = {f"level_{lev}": float(torch.stack(descs).norm(dim=-1).mean()) for lev, descs in tree.items()}
    return norms


def ablation_tables() -> dict[str, Any]:
    return {
        "table2_hypevpr_l_r1": TABLE2_PITTS["HypeVPR-L_star"]["r1"],
        "table2_eigenplace_r1": TABLE2_PITTS["EigenPlace_star"]["r1"],
        "table4_poincare_beats_euclidean": TABLE4_MANIFOLD["Poincare"]["r1"]
        > TABLE4_MANIFOLD["Euclidean"]["r1"],
    }
