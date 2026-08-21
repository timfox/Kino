"""LTX / native-evolve integration for FUSE-Flow geometry-aware training."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fuse_flow.config import FuseFlowConfig


def ltx_integration_plan(cfg: FuseFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FuseFlowConfig()
    return {
        "goal": "Metric-consistent multi-view geometry in latent training (beyond public LTX)",
        "phases": [
            {
                "id": "fold_backfill",
                "action": "GOPEX_ENABLE_AV_FOLD=1 + fuse_flow in GOPEX_AV_FOLD_HOOKS",
                "tool": "./scripts/kino-fuse-flow-ltx.sh backfill",
                "artifact": "latents/*.pt fuse_flow sidecars",
            },
            {
                "id": "train_weights",
                "action": "av_fold.use_fuse_flow_weights + downweight_low_geometry_stability",
                "env": "GOPEX_AV_FOLD_TRAIN=1 (default in native yaml)",
                "effect": "Upweight fusion-ready clips; downweight unstable geometry",
            },
            {
                "id": "native_evolve",
                "action": "Continue merged_native connector + HDR ladder after plus_gphotos",
                "tool": "./scripts/kino-native-evolve.sh train-connectors",
            },
            {
                "id": "depth_export",
                "action": "Per-view depth PNG + GMAC scale lift → conditioning maps (future GPU)",
                "tool": "fuse_flow eval_demo / upstream GMAC release",
            },
        ],
        "recommended_hooks": [
            "fuse_flow",
            "phyworld",
            "sphere_depth",
            "pantheon360",
            "avbench",
        ],
        "use_cases": [
            "Multi-camera volumetric capture QA for CR17 / teleplay sets",
            "Expansion datasets (pd-horror-movies) with rig metadata",
            "Extrinsic drift monitoring on synchronized camera arrays",
        ],
        "complexity": cfg.complexity_per_frame,
        "train_fields": [
            "geometry_stability_proxy",
            "fusion_readiness_proxy",
            "multi_view_hint",
        ],
    }


def native_evolve_env_snippet() -> str:
    """Shell exports for fold + train without editing yaml by hand."""
    return "\n".join(
        [
            "export GOPEX_ENABLE_AV_FOLD=1",
            "export GOPEX_AV_FOLD_HOOKS=fuse_flow,phyworld,sphere_depth,pantheon360,avbench",
            "export GOPEX_AV_FOLD_TRAIN=1",
        ]
    )
