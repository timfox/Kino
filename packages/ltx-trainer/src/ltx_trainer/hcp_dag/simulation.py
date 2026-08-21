"""Simulation: boundary-shift stability vs algorithm [3] (Sec. III)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hcp_dag.config import HCPDAGConfig
from ltx_trainer.hcp_dag.graph import build_crack_chains, build_crack_dag
from ltx_trainer.hcp_dag.reference import boundary_shifted_graph, yang_fig4a_toy_graph
from ltx_trainer.hcp_dag.topology import classify_crack_generations, generation_stability_score, peel_stages


def run_on_graph(graph, *, cfg: HCPDAGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HCPDAGConfig()
    chains = build_crack_chains(graph, cfg.rules)
    dag = build_crack_dag(chains, graph)
    leaf = classify_crack_generations(dag, method="leaf_peel")
    root = classify_crack_generations(dag, method="root_peel")
    return {
        "n_cracks": dag["n_cracks"],
        "n_generations_leaf": leaf["n_generations"],
        "generations_leaf": leaf["generations"],
        "generations_root": root["generations"],
        "dag": dag,
        "chains": chains,
    }


def crack_node_signature(chains: list[list[int]], graph) -> list[frozenset[tuple[int, int]]]:
    """Identify cracks by sorted node pairs (stable across boundary crops)."""
    sigs: list[frozenset[tuple[int, int]]] = []
    for chain in chains:
        pairs: set[tuple[int, int]] = set()
        for eidx in chain:
            e = graph.edges[eidx]
            pairs.add(tuple(sorted((e.u, e.v))))
        sigs.append(frozenset(pairs))
    return sigs


def match_cracks(
    full_sigs: list[frozenset[tuple[int, int]]],
    frag_sigs: list[frozenset[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Pair crack ids with identical or subset node-pair sets."""
    pairs: list[tuple[int, int]] = []
    used_frag: set[int] = set()
    for i, sig in enumerate(full_sigs):
        for j, sig2 in enumerate(frag_sigs):
            if j in used_frag:
                continue
            if sig == sig2 or sig <= sig2 or sig2 <= sig:
                pairs.append((i, j))
                used_frag.add(j)
                break
    return pairs


def _stability_vs_full(
    full: dict[str, Any],
    full_g,
    frag_g,
    *,
    gen_key: str,
) -> dict[str, Any]:
    frag = run_on_graph(frag_g)
    matched = match_cracks(
        crack_node_signature(full["chains"], full_g),
        crack_node_signature(frag["chains"], frag_g),
    )
    gen_full = {fi: full[gen_key][fi] for fi, _ in matched}
    gen_frag = {fi: frag[gen_key][lj] for fi, lj in matched}
    return {
        "matched_cracks": len(matched),
        "stability": generation_stability_score(gen_full, gen_frag, set(gen_full)),
        "fragment_generations": frag[gen_key],
    }


def boundary_shift_stability_demo(cfg: HCPDAGConfig | None = None) -> dict[str, Any]:
    """Fig. 4–5 analogue: leaf-peel stable, root-peel [3] unstable."""
    cfg = cfg or HCPDAGConfig()
    full_g = yang_fig4a_toy_graph()
    full = run_on_graph(full_g, cfg=cfg)

    left_g = boundary_shifted_graph(drop_left=True)
    right_g = boundary_shifted_graph(drop_right=True)
    left = run_on_graph(left_g, cfg=cfg)

    matched_left = match_cracks(
        crack_node_signature(full["chains"], full_g),
        crack_node_signature(left["chains"], left_g),
    )
    gen_full_leaf = {fi: full["generations_leaf"][fi] for fi, _ in matched_left}
    gen_left_leaf = {fi: left["generations_leaf"][lj] for fi, lj in matched_left}
    leaf_stability = generation_stability_score(gen_full_leaf, gen_left_leaf, set(gen_full_leaf))

    gen_full_root = {fi: full["generations_root"][fi] for fi, _ in matched_left}
    gen_left_root = {fi: left["generations_root"][lj] for fi, lj in matched_left}
    root_stability = generation_stability_score(gen_full_root, gen_left_root, set(gen_full_root))

    right_leaf = _stability_vs_full(full, full_g, right_g, gen_key="generations_leaf")
    right_root = _stability_vs_full(full, full_g, right_g, gen_key="generations_root")

    return {
        "full_n_generations": full["n_generations_leaf"],
        "left_fragment_n_generations": left["n_generations_leaf"],
        "matched_cracks": len(matched_left),
        "leaf_peel_stability": leaf_stability,
        "root_peel_stability": root_stability,
        "leaf_peel_stability_right": right_leaf["stability"],
        "root_peel_stability_right": right_root["stability"],
        "leaf_more_stable_than_root": leaf_stability > root_stability,
        "full_generations_leaf": full["generations_leaf"],
        "left_generations_leaf": left["generations_leaf"],
        "full_generations_root": full["generations_root"],
        "left_generations_root": left["generations_root"],
        "right_generations_leaf": right_leaf["fragment_generations"],
    }


def peel_stages_demo(cfg: HCPDAGConfig | None = None) -> dict[str, Any]:
    """Fig. 2 vs Fig. 3: leaf-peel vs root-peel stage sequences."""
    cfg = cfg or HCPDAGConfig()
    graph = yang_fig4a_toy_graph()
    run = run_on_graph(graph, cfg=cfg)
    dag = run["dag"]
    return {
        "leaf_peel_stages": peel_stages(dag, method="leaf_peel"),
        "root_peel_stages": peel_stages(dag, method="root_peel"),
    }


def full_pipeline_demo(cfg: HCPDAGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HCPDAGConfig()
    main = run_on_graph(yang_fig4a_toy_graph(), cfg=cfg)
    stability = boundary_shift_stability_demo(cfg=cfg)
    stages = peel_stages_demo(cfg=cfg)
    return {
        "reference_graph": "yang_fig4a_toy",
        "n_cracks": main["n_cracks"],
        "n_generations": main["n_generations_leaf"],
        "youngest_cracks": [c for c, g in main["generations_leaf"].items() if g == 0],
        "boundary_shift": stability,
        "peel_stages": stages,
    }
