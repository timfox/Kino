"""ForestHG-Trace framework API (arXiv:2605.27590)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.foresthg_trace.benchmarks import benchmarks_bundle
from ltx_trainer.foresthg_trace.config import ForestHGConfig
from ltx_trainer.foresthg_trace.examples import build_demo_scene, gold_slope_ratio_program
from ltx_trainer.foresthg_trace.execution import run_program
from ltx_trainer.foresthg_trace.metrics import answer_match, trace_coverage, trace_similarity


def framework_card(cfg: ForestHGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ForestHGConfig()
    return {
        "name": "ForestHG-Trace",
        "paper": f"arXiv:{cfg.arxiv}",
        "representation": "multimodal ecological scene hypergraph H=(V,E,X)",
        "hyperedge_families": list(cfg.hyperedge_families),
        "operators": list(cfg.operators),
        "benchmark": "ForestTraceQA",
        "neon_scenes": f"{cfg.neon_sites} sites × {cfg.tiles_per_site} tiles = {cfg.num_scenes}",
        "benchmark_instances": cfg.benchmark_instances,
        "task_groups": list(cfg.task_groups),
        "reasoning_levels": list(cfg.reasoning_levels),
        "modalities": list(cfg.modalities),
        "default_backbone": cfg.default_backbone,
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.27590",
        "authors": "Cheng, Wang, Li, Huang, Fu, Wang (XJTU, Xidian, ZJU)",
        "problem": "RS-QA over forests needs executable multi-step analysis, not one-shot VLM answers",
        "framework": [
            "Scene hypergraph with BEH/REH/CEH hyperedge families",
            "Deterministic tool operators: read, filter, expand, aggregate, compare, audit",
            "LLM plans; tools execute with replayable traces",
        ],
        "benchmark": {
            "name": "ForestTraceQA",
            "axes": ["execution difficulty H0–H4", "task type BP/SA/IA/MT/PO"],
            "records": "program_dag, tool_trace, intermediate_states, final_answer",
        },
        "headline_result": "54.45% overall accuracy vs 31.01% scene-graph agent",
        "limitation": "Performance drops sharply past ~50 tool calls; execution-limited not perception-limited",
    }


def evaluation_demo(seed: int = 0) -> dict[str, Any]:
    graph = build_demo_scene(seed)
    program = gold_slope_ratio_program()
    trace = run_program(
        graph,
        program,
        evidence={
            "question": "Which slope has a higher poor-health tree ratio?",
            "task_group": "SA",
            "level": "H2",
        },
    )
    gold_answer = "Shady slope has a higher poor-health tree ratio."
    pred_ops = trace.tool_trace
    gold_ops = [c.op for c in program]
    return {
        "scene_id": graph.scene_id,
        "tree_count": len(graph.nodes),
        "hyperedge_count": len(graph.hyperedges),
        "question": trace.evidence.get("question"),
        "predicted_answer": trace.answer,
        "gold_answer": gold_answer,
        "answer_correct": answer_match(trace.answer, gold_answer),
        "trace_coverage": trace_coverage(pred_ops, gold_ops),
        "trace_similarity": trace_similarity(pred_ops, gold_ops),
        "tool_trace": pred_ops,
        "final_scalars": trace.intermediate_states[-2]["result"] if len(trace.intermediate_states) >= 2 else {},
        "record": trace.to_record(),
    }


def evaluation_smoke(cfg: ForestHGConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo(seed=0)
    return {
        "paper": "arXiv:2605.27590",
        "answer_correct": demo["answer_correct"],
        "trace_coverage": demo["trace_coverage"],
        "trace_similarity": demo["trace_similarity"],
        "tree_count": demo["tree_count"],
        "ok": demo["answer_correct"] and demo["trace_similarity"] >= 0.9,
    }
