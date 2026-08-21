"""Framework card, demos, smoke for HCP DAG crack generations (arXiv:2606.03473)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hcp_dag.config import HCPDAGConfig
from ltx_trainer.hcp_dag.paper_tables import (
    algorithm_steps,
    comparison_algorithms,
    knowledge_card,
    reference_anchors,
)
from ltx_trainer.hcp_dag.simulation import full_pipeline_demo


def framework_card(cfg: HCPDAGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HCPDAGConfig()
    anchors = reference_anchors()
    return {
        "name": "HCP-DAG",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "title": "Hierarchical crack patterns: identification of crack generations",
        "task": "Reduce crack-generation labeling to topological sort of a child→parent DAG",
        "pipeline": algorithm_steps(),
        "rules": {
            "chain_angle_deg": [cfg.rules.min_chain_angle_deg, cfg.rules.max_chain_angle_deg],
            "max_vertex_degree": cfg.rules.max_vertex_degree,
        },
        "comparison": comparison_algorithms(),
        "reference_anchors": anchors,
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub on synthetic toy graphs — no image digitization or skeletonization.",
        "Requires hierarchical network (deg ≤ 3, no loops); X/Y junctions not handled.",
        "Boundary-shift demo uses edge-set matching on a minimal reference graph.",
        "Crack width / temporal hierarchy not incorporated (Sec. IV future work).",
    ]


def evaluation_demo(cfg: HCPDAGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HCPDAGConfig()
    demo = full_pipeline_demo(cfg=cfg)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "knowledge": knowledge_card(),
        "reference_anchors": reference_anchors(),
        "pipeline_demo": demo,
    }


def evaluation_smoke(cfg: HCPDAGConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    cfg = cfg or HCPDAGConfig()
    pipe = demo["pipeline_demo"]
    anchors = demo["reference_anchors"]
    shift = pipe["boundary_shift"]

    assert pipe["n_cracks"] == anchors["yang_fig4a_toy_n_cracks"]
    assert pipe["n_generations"] == anchors["yang_fig4a_toy_n_generations"]
    assert len(pipe["youngest_cracks"]) >= 2
    assert shift["leaf_peel_stability"] >= anchors["leaf_peel_boundary_stability_min"]
    assert shift["leaf_more_stable_than_root"] is True
    assert shift["matched_cracks"] >= 2

    return {
        "status": "ok",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "n_cracks": pipe["n_cracks"],
        "n_generations": pipe["n_generations"],
        "leaf_peel_stability": shift["leaf_peel_stability"],
        "root_peel_stability": shift["root_peel_stability"],
        "demo_keys": list(demo.keys()),
    }
