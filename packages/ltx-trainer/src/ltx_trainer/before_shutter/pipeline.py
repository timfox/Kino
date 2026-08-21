"""Framework card, benchmark manifest, and smoke demos (arXiv:2605.30318)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.before_shutter.camera import ev100_from_aperture_shutter, exposure_validity_logit, render_linear_radiance
from ltx_trainer.before_shutter.config import BeforeShutterConfig, PortraitPlanState
from ltx_trainer.before_shutter.human import batch_actionability
from ltx_trainer.before_shutter.lighting import apply_preset, devices_from_state, lighting_ratio
from ltx_trainer.before_shutter.metrics import bradley_terry_win_probability, table1_main_results, table2_stage_ablation
from ltx_trainer.before_shutter.planning import run_comparative_planning
from ltx_trainer.before_shutter.scene_graph import build_demo_graph, probe_emitter_ratios


def framework_card(cfg: BeforeShutterConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BeforeShutterConfig()
    return {
        "name": "Before the Shutter",
        "paper": cfg.paper_arxiv,
        "repo": cfg.repo_url,
        "task": "3D aesthetic portrait planning before capture",
        "roles": ["Photographer", "Actor", "Judge"],
        "planning_stages": ["staging", "composition", "lighting"],
        "representation": "Photographic Scene Graph (V_non, V_emi, E_n2n, E_e2n, E_e2e)",
        "algorithm": "Aesthetic-Guided Comparative Planning with aesthetic frontier F_t",
        "controls": [
            "SMPL-X pose + root transform (H)",
            "Thin-lens camera T, K, aperture-priority P (C)",
            "Controllable lights + exposure compensation (L)",
        ],
        "benchmark": {
            "tasks": cfg.benchmark_tasks,
            "scenes": cfg.benchmark_scenes,
            "indoor": cfg.benchmark_indoor,
            "outdoor": cfg.benchmark_outdoor,
        },
        "implementation_notes": [
            "Virtual env: Blender Cycles; real scenes via occupancy + radiance queries",
            "Photographer/Judge: MLLM agents (GPT-5.4 family in paper)",
            "Staging: 2D pose prior + SMPLer-X lift + Actor constraints",
        ],
    }


def paper_limitations() -> list[str]:
    return [
        "Facial expression and gaze fixed in paper experiments (uncanny valley on realistic faces).",
        "Per-task runtime 5–25 min bounded by Cycles rendering.",
        "Requires static 3D scene with occupancy O and radiance L(o, ω_o).",
        "Full upstream release: github.com/songrise/Before-the-Shutter",
    ]


def benchmark_manifest(cfg: BeforeShutterConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or BeforeShutterConfig()
    prompts = [
        "Graceful pianist, thoughtful",
        "Melancholy",
        "Mafia, dramatic chiaroscuro",
        "Confident executive, minimalist interior",
        "Resting on pier, solemn",
        "Horrifying mood, low key",
    ]
    scenes = [
        "studio_neutral",
        "library_window",
        "pagoda_outdoor",
        "glass_wall_dance",
        "pier_sunset",
        "dark_hallway",
    ]
    out: list[dict[str, Any]] = []
    for i in range(min(cfg.benchmark_tasks, 12)):
        out.append(
            {
                "task_id": i + 1,
                "scene": scenes[i % len(scenes)],
                "prompt": prompts[i % len(prompts)],
                "indoor": i % 3 != 2,
            }
        )
    return out


def table_baselines() -> list[str]:
    return [row["method"] for row in table1_main_results()]


def evaluation_demo(*, cfg: BeforeShutterConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BeforeShutterConfig()
    graph = build_demo_graph(prompt="Melancholy")
    state = apply_preset(PortraitPlanState(), "rembrandt")
    devices = devices_from_state(state)
    ev = ev100_from_aperture_shutter(state.f_number, 1.0 / 125.0)
    base = torch.full((128, 128), 0.4)
    img = render_linear_radiance(base, ev100=ev, exposure_comp_stops=state.exposure_comp_stops)
    p_valid, v_exp = exposure_validity_logit(img, cfg=cfg)
    ratios = probe_emitter_ratios(
        meter_amb=0.07,
        meter_with_emitter={"key": 0.5, "fill": 0.12},
        distances_m={"key": 1.0, "fill": 1.5},
        cfg=cfg,
    )
    plan = run_comparative_planning(prompt="Melancholy, sit on red chair", cfg=cfg, seed=42)
    act = batch_actionability([state], cfg=cfg)
    bt = bradley_terry_win_probability(1.30, 1.07)
    return {
        "graph_nodes": len(graph.nodes_non) + len(graph.nodes_emi),
        "lighting_ratio_key_fill": round(lighting_ratio(state.light_powers_w[0], state.light_powers_w[1]), 2),
        "num_light_devices": len(devices),
        "ev100": round(ev, 3),
        "p_valid": round(p_valid, 4),
        "V_exp": round(v_exp, 4),
        "emitter_ratios": len(ratios),
        "planning": plan,
        "actionability": act,
        "bt_ours_vs_greedy": round(bt, 4),
    }


def training_step_demo(*, cfg: BeforeShutterConfig | None = None) -> dict[str, Any]:
    """Training-free system — returns one planning trace."""
    return run_comparative_planning(cfg=cfg or BeforeShutterConfig(), prompt="Thoughtful, pianist")
