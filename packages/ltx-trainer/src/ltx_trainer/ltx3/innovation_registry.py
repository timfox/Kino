"""LTX 3.0 innovation registry — maps research papers → pillars → hooks → train/eval.

This is the curated product view over ``fold_registry`` (78+ sidecars) and deep
integrations (Bernini, physics steering, VCap, ID-LoRA, minute compose, HDR ladder).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ltx_trainer.fold_registry import list_registered_hooks


class Pillar(str, Enum):
    """Four LTX 3.0 capability pillars (see documents/LTX_3_VISION.md)."""

    LONG_CONDITIONING = "long_conditioning"
    MINUTE_AV = "minute_av"
    QUALITY_LOOP = "quality_loop"
    PAPER_HOOKS = "paper_hooks"


class Tier(str, Enum):
    """How deeply an innovation is wired into Kino today."""

    METADATA = "metadata"  # fold sidecar on .pt shards
    TRAIN_GATE = "train_gate"  # av_fold loss / sample weight
    INFER = "infer"  # guidance / steering at sample time
    COMPOSE = "compose"  # minute planner / render driver
    PRODUCT = "product"  # bundle export / gate / orchestrator


@dataclass(frozen=True)
class Innovation:
    id: str
    title: str
    pillar: Pillar
    tier: Tier
    doc: str
    hook: str | None = None
    gate: str | None = None
    phase: str | None = None
    command: str | None = None
    module: str | None = None


# Curated LTX 3.0 stack — deep + train-gate papers (not exhaustive metadata list).
_CURATED: tuple[Innovation, ...] = (
    Innovation(
        "gemma4_native",
        "Gemma 4 native fold + connectors",
        Pillar.LONG_CONDITIONING,
        Tier.PRODUCT,
        "documents/LTX_WEIGHTS.md",
        gate="G1",
        phase="A",
        command="./scripts/kino-native-evolve.sh train-connectors",
        module="ltx_trainer.gemma4",
    ),
    Innovation(
        "bernini",
        "Bernini latent semantic segment planner",
        Pillar.LONG_CONDITIONING,
        Tier.TRAIN_GATE,
        "documents/BERNINI.md",
        hook="bernini",
        phase="B",
        module="ltx_trainer.bernini",
    ),
    Innovation(
        "vcap",
        "VCap witness-adjudicator caption RL",
        Pillar.LONG_CONDITIONING,
        Tier.TRAIN_GATE,
        "documents/VCAP.md",
        phase="A",
        command="./scripts/gopex-vcap.sh smoke",
        module="ltx_trainer.vcap",
    ),
    Innovation(
        "longav_compass",
        "LongAV-Compass minute-scale eval",
        Pillar.MINUTE_AV,
        Tier.TRAIN_GATE,
        "documents/LONGAV_COMPASS.md",
        hook="longav_compass",
        gate="G2",
        phase="B",
        command="./scripts/gopex-longav-compass.sh ltx-plan",
        module="ltx_trainer.longav_compass",
    ),
    Innovation(
        "minute_compose",
        "Composed minute AV (hero carry + concat)",
        Pillar.MINUTE_AV,
        Tier.COMPOSE,
        "documents/LTX_3_VISION.md",
        gate="G2",
        phase="B",
        command="./scripts/kino-ltx3-minute-render.sh render-plan --longav",
        module="ltx_trainer.ltx3.minute_compose",
    ),
    Innovation(
        "mtavg2",
        "MTAVG-Bench cinematic failure diagnosis",
        Pillar.MINUTE_AV,
        Tier.TRAIN_GATE,
        "documents/MTAVG_BENCH2.md",
        hook="mtavg2",
        phase="B",
        command="./scripts/gopex-mtavg2.sh smoke",
        module="ltx_trainer.mtavg2",
    ),
    Innovation(
        "avbench",
        "AVBench human-aligned T2AV metrics",
        Pillar.QUALITY_LOOP,
        Tier.TRAIN_GATE,
        "documents/AVBENCH.md",
        hook="avbench",
        gate="G1",
        command="./scripts/kino eval",
        module="ltx_trainer.avbench",
    ),
    Innovation(
        "adamag",
        "AdaMaG manifold guidance (HQ infer)",
        Pillar.QUALITY_LOOP,
        Tier.INFER,
        "documents/ADAMAG.md",
        hook="adamag",
        gate="G1",
        module="ltx_trainer.adamag",
    ),
    Innovation(
        "vsr_vqa",
        "VSR-VQA diffusion quality weights",
        Pillar.QUALITY_LOOP,
        Tier.TRAIN_GATE,
        "documents/VSR_VQA.md",
        hook="vsr_vqa",
        module="ltx_trainer.vsr_vqa",
    ),
    Innovation(
        "phyworld",
        "PhyWorld physics-faithful world model aux",
        Pillar.QUALITY_LOOP,
        Tier.TRAIN_GATE,
        "documents/PHYWORLD.md",
        hook="phyworld",
        module="ltx_trainer.phyworld",
    ),
    Innovation(
        "physics_steering",
        "PEZ CAV physics steering at infer",
        Pillar.QUALITY_LOOP,
        Tier.INFER,
        "documents/PHYSICS_STEERING.md",
        command="./scripts/gopex-physics-steering.sh synthetic",
        module="ltx_trainer.physics_steering",
    ),
    Innovation(
        "fuse_flow",
        "FUSE-Flow metric geometry sidecar",
        Pillar.QUALITY_LOOP,
        Tier.TRAIN_GATE,
        "documents/FUSE_FLOW.md",
        hook="fuse_flow",
        command="./scripts/kino-fuse-flow-ltx.sh backfill",
        module="ltx_trainer.fuse_flow",
    ),
    Innovation(
        "lamo",
        "LaMo motion drift regularizer",
        Pillar.QUALITY_LOOP,
        Tier.TRAIN_GATE,
        "documents/LAMO.md",
        hook="lamo",
        module="ltx_trainer.lamo",
    ),
    Innovation(
        "latenthdr",
        "LatentHDR exposure head + LogC3 ladder",
        Pillar.QUALITY_LOOP,
        Tier.TRAIN_GATE,
        "documents/LATENTHDR.md",
        hook="latenthdr",
        gate="G3",
        phase="C",
        command="./scripts/kino-native-evolve.sh hdr-ladder",
        module="ltx_trainer.latenthdr",
    ),
    Innovation(
        "pd_horror_g4",
        "Catalog growth gate (pd-horror → merged_native)",
        Pillar.QUALITY_LOOP,
        Tier.PRODUCT,
        "documents/LTX_3_VISION.md",
        gate="G4",
        phase="now",
        command="./scripts/kino pd-horror plan",
        module="tools.pd_horror_g4",
    ),
    Innovation(
        "id_lora",
        "ID-LoRA + audio_ref_only_ic cross-modal masking",
        Pillar.PAPER_HOOKS,
        Tier.TRAIN_GATE,
        "documents/ID_LORA.md",
        hook="id_lora",
        command="./scripts/kino-id-lora.sh verify",
        module="ltx_trainer.id_lora",
    ),
    Innovation(
        "safedig",
        "SafeDIG DiT safety steering",
        Pillar.PAPER_HOOKS,
        Tier.INFER,
        "documents/SAFEDIG.md",
        command="./scripts/gopex-safedig.sh smoke",
        module="ltx_trainer.safedig",
    ),
    Innovation(
        "omnicustom",
        "OmniCustom sync AV customization",
        Pillar.PAPER_HOOKS,
        Tier.TRAIN_GATE,
        "documents/OMNICUSTOM.md",
        hook="omnicustom",
        module="ltx_trainer.omnicustom",
    ),
    Innovation(
        "autocut",
        "AutoCut ad edit discretization",
        Pillar.PAPER_HOOKS,
        Tier.COMPOSE,
        "documents/AUTOCUT.md",
        hook="autocut",
        command="./scripts/gopex-autocut.sh edit --scenario script_driven",
        module="ltx_trainer.autocut",
    ),
    Innovation(
        "swansphere",
        "SwanSphere spatial audio (mel proxy → future VAE)",
        Pillar.PAPER_HOOKS,
        Tier.METADATA,
        "documents/SWANSPHERE.md",
        hook="swansphere_audio",
        phase="E",
        module="ltx_trainer.swansphere",
    ),
    Innovation(
        "foley_omni",
        "Foley-Omni spatial foley alignment",
        Pillar.PAPER_HOOKS,
        Tier.METADATA,
        "documents/FOLEY_OMNI.md",
        hook="foley_omni_audio",
        phase="E",
        module="ltx_trainer.foley_omni",
    ),
    Innovation(
        "temporal_upscale",
        "LTX temporal upscaler x2 (minute path)",
        Pillar.MINUTE_AV,
        Tier.INFER,
        "documents/LTX_STUBS.md",
        phase="D",
        command="./scripts/kino-ltx3-evolve.sh temporal-status",
        module="ltx_trainer.ltx3.temporal_upscale",
    ),
    Innovation(
        "sphere360",
        "Sphere360 ERP dataset + fold",
        Pillar.PAPER_HOOKS,
        Tier.METADATA,
        "documents/SPHERE360.md",
        hook="sphere360",
        module="gopex_datasets.sphere360",
    ),
    Innovation(
        "pantheon360",
        "Pantheon360 3D-aware 360° video",
        Pillar.PAPER_HOOKS,
        Tier.METADATA,
        "documents/PANTHEON360.md",
        hook="pantheon360",
        module="ltx_trainer.pantheon360",
    ),
    Innovation(
        "mirage",
        "Mirage latent spatial memory for video world models",
        Pillar.PAPER_HOOKS,
        Tier.TRAIN_GATE,
        "documents/MIRAGE.md",
        hook="mirage",
        module="ltx_trainer.mirage",
    ),
    Innovation(
        "flatsounds",
        "FlatSounds V2A physical benchmark",
        Pillar.PAPER_HOOKS,
        Tier.TRAIN_GATE,
        "documents/FLATSOUNDS.md",
        hook="flatsounds",
        module="ltx_trainer.flatsounds",
    ),
    Innovation(
        "growloop",
        "GrowLoop human-likeness eval loop",
        Pillar.QUALITY_LOOP,
        Tier.METADATA,
        "documents/GROWLOOP.md",
        hook="growloop",
        module="ltx_trainer.growloop",
    ),
    Innovation(
        "ltx23_hdr_ic_lora",
        "Official LTX-2.3 HDR IC-LoRA control asset",
        Pillar.QUALITY_LOOP,
        Tier.INFER,
        "documents/LTX_3_VISION.md",
        hook="latenthdr",
        gate="G3",
        phase="C",
        command="./scripts/kino-native-evolve.sh hdr-ladder",
        module="ltx_trainer.latenthdr",
    ),
    Innovation(
        "ltx23_motion_track_control",
        "Official LTX-2.3 motion-track IC-LoRA control asset",
        Pillar.MINUTE_AV,
        Tier.INFER,
        "documents/LTX_3_VISION.md",
        hook="semantic_stitch",
        gate="G2",
        phase="B",
        command="./scripts/kino-ltx3-minute-render.sh render-plan --longav",
        module="ltx_trainer.ltx3.minute_compose",
    ),
    Innovation(
        "video_weave_geometry",
        "VideoWeave-style explicit geometry consistency",
        Pillar.QUALITY_LOOP,
        Tier.TRAIN_GATE,
        "documents/FUSE_FLOW.md",
        hook="fuse_flow",
        gate="G1",
        phase="A",
        command="./scripts/kino-fuse-flow-ltx.sh backfill",
        module="ltx_trainer.fuse_flow",
    ),
    Innovation(
        "memory_bank_long_video",
        "MDiT-style memory bank for long-video identity continuity",
        Pillar.MINUTE_AV,
        Tier.TRAIN_GATE,
        "documents/MIRAGE.md",
        hook="mirage",
        gate="G2",
        phase="B",
        command="./scripts/kino-ltx3-evolve.sh minute-plan",
        module="ltx_trainer.mirage",
    ),
    Innovation(
        "fast_ar_rolling_forcing",
        "Rolling-forcing long-video cache discipline",
        Pillar.MINUTE_AV,
        Tier.INFER,
        "documents/LTX_3_VISION.md",
        gate="G2",
        phase="D",
        command="./scripts/kino-ltx3-evolve.sh temporal-status",
        module="ltx_trainer.ltx3.temporal_upscale",
    ),
    Innovation(
        "plan_verify_reward",
        "Plan-and-verify video reward reasoning gate",
        Pillar.QUALITY_LOOP,
        Tier.TRAIN_GATE,
        "documents/VCAP.md",
        gate="G1",
        phase="A",
        command="./scripts/kino-native-evolve.sh eval",
        module="ltx_trainer.vcap",
    ),
    Innovation(
        "reward_gradient_alignment",
        "Video diffusion reward-gradient alignment loop",
        Pillar.QUALITY_LOOP,
        Tier.TRAIN_GATE,
        "documents/VCAP.md",
        gate="G4",
        phase="A",
        command="./scripts/kino-native-evolve.sh eval",
        module="ltx_trainer.vcap",
    ),
)


@dataclass
class InnovationStack:
    """Preset bundle of hooks + env for an LTX 3 evolution track."""

    name: str
    description: str
    pillars: list[Pillar]
    hooks: list[str]
    env: dict[str, str] = field(default_factory=dict)
    train_profile: str = "ltx3_quality"
    gates: list[str] = field(default_factory=lambda: ["G1", "G2", "G3", "G4"])
    commands: list[str] = field(default_factory=list)


STACKS: dict[str, InnovationStack] = {
    "ltx3_full": InnovationStack(
        name="ltx3_full",
        description="All four pillars: long prompts, minute compose, quality train gates, full fold sidecars.",
        pillars=list(Pillar),
        hooks=[],  # filled at runtime from fold_registry default preprocess list
        env={
            "GOPEX_ENABLE_AV_FOLD": "1",
            "GOPEX_AV_FOLD_TRAIN": "1",
            "GOPEX_RESEARCH_PROFILE": "ltx3_quality",
            "LTX_GEMMA_ENCODE_CAP": "8192",
            "GOPEX_LTX3_HQ": "1",
            "GOPEX_LTX3_PLAN_VERIFY_REWARD": "1",
            "GOPEX_LTX3_REWARD_GRADIENTS": "1",
            "GOPEX_LTX3_GEOMETRY_WEAVE": "1",
            "GOPEX_LTX3_MEMORY_BANK": "1",
            "GOPEX_LTX3_ROLLING_FORCING": "1",
            "GOPEX_LTX23_IC_LORA_SCAN": "1",
            "GOPEX_LTX23_HDR_IC_LORA": "1",
            "GOPEX_LTX23_MOTION_TRACK_CONTROL": "1",
        },
        commands=[
            "./scripts/kino-ltx3-evolve.sh plan",
            "./scripts/kino-native-evolve.sh train-connectors",
            "./scripts/kino-ltx3-minute-render.sh dry-run --longav",
            "./scripts/kino-native-evolve.sh hdr-ladder",
        ],
    ),
    "ltx3_minute": InnovationStack(
        name="ltx3_minute",
        description="Minute composed T2AV + LongAV/MTAVG eval (G2 focus).",
        pillars=[Pillar.LONG_CONDITIONING, Pillar.MINUTE_AV],
        hooks=[
            "bernini",
            "longav_compass",
            "mtavg2",
            "avbench",
            "lamo",
            "adamag",
            "flatsounds",
        ],
        env={
            "GOPEX_ENABLE_AV_FOLD": "1",
            "GOPEX_AV_FOLD_TRAIN": "1",
            "GOPEX_RESEARCH_PROFILE": "ltx3_quality",
            "LTX_GEMMA_ENCODE_CAP": "8192",
            "GOPEX_LTX3_HQ": "1",
            "GOPEX_LTX3_MEMORY_BANK": "1",
            "GOPEX_LTX3_ROLLING_FORCING": "1",
            "GOPEX_LTX23_MOTION_TRACK_CONTROL": "1",
        },
        gates=["G2"],
        commands=[
            "./scripts/kino-ltx3-evolve.sh minute-longav",
            "./scripts/kino-ltx3-minute-render.sh render-plan --longav",
            "./scripts/gopex-longav-compass.sh ltx-plan",
        ],
    ),
    "ltx3_hdr": InnovationStack(
        name="ltx3_hdr",
        description="HDR ladder + panorama/HDR fold sidecars (G3 focus).",
        pillars=[Pillar.QUALITY_LOOP, Pillar.PAPER_HOOKS],
        hooks=[
            "latenthdr",
            "lucky_hdr",
            "sdr2hdr",
            "diffhdr",
            "lumivid",
            "stem2_sdr_hdr",
            "modulo_spike_hdr",
            "physthdr_gs",
        ],
        env={
            "GOPEX_HDR_PREPROCESS": "1",
            "GOPEX_ENABLE_AV_FOLD": "1",
            "GOPEX_RESEARCH_PROFILE": "ltx3_quality",
            "GOPEX_LTX23_HDR_IC_LORA": "1",
            "GOPEX_LTX23_IC_LORA_SCAN": "1",
        },
        gates=["G3"],
        commands=["./scripts/kino-native-evolve.sh hdr-ladder"],
    ),
    "ltx3_audio": InnovationStack(
        name="ltx3_audio",
        description="Speech/audio alignment stack (SwanSphere proxy, codec aux, ID-LoRA).",
        pillars=[Pillar.PAPER_HOOKS, Pillar.QUALITY_LOOP],
        hooks=[
            "fmelcodec",
            "speech_quality_emb",
            "robustspeechflow",
            "wavenext2",
            "ag_repa",
            "swansphere_audio",
            "foley_omni_audio",
            "id_lora",
            "id_lora_audio",
            "omnicustom",
            "omnicustom_audio",
            "avbench",
        ],
        env={
            "GOPEX_ENABLE_AV_FOLD": "1",
            "GOPEX_AV_FOLD_TRAIN": "1",
            "GOPEX_AUDIO_ALIGN_VIDEO": "1",
            "GOPEX_RESEARCH_PROFILE": "speech",
        },
        commands=["./scripts/kino-id-lora.sh verify", "./scripts/gopex-physics-steering.sh synthetic"],
    ),
    "ltx3_panorama": InnovationStack(
        name="ltx3_panorama",
        description="360° / ERP research fold cluster for sphere-native training.",
        pillars=[Pillar.PAPER_HOOKS],
        hooks=[
            "sphere360",
            "pantheon360",
            "dense360",
            "panoworld",
            "gimbal360",
            "sphere_depth",
            "nvc_erp_qpa",
            "fuse_flow",
        ],
        env={
            "GOPEX_ENABLE_AV_FOLD": "1",
            "GOPEX_RESEARCH_PROFILE": "panorama",
        },
        commands=["./scripts/kino-sphere360-ltx-plan.sh plan"],
    ),
    "ltx3_world_model": InnovationStack(
        name="ltx3_world_model",
        description="Latent spatial memory + physics world-model train gates (Mirage, PhyWorld, FUSE-Flow).",
        pillars=[Pillar.PAPER_HOOKS, Pillar.QUALITY_LOOP],
        hooks=[
            "mirage",
            "phyworld",
            "fuse_flow",
            "avbench",
            "longav_compass",
            "bernini",
            "lamo",
        ],
        env={
            "GOPEX_ENABLE_AV_FOLD": "1",
            "GOPEX_AV_FOLD_TRAIN": "1",
            "GOPEX_RESEARCH_PROFILE": "world_model",
            "GOPEX_AV_FOLD_HOOKS": "mirage,phyworld,fuse_flow,avbench,longav_compass",
        },
        train_profile="world_model",
        gates=["G1", "G4"],
        commands=[
            "./scripts/kino-mirage.sh verify",
            "./scripts/gopex-mirage.sh smoke",
            "eval \"$(python tools/ltx_research_profile.py env world_model)\"",
            "./scripts/kino-research-profile.sh patch-configs world_model",
        ],
    ),
}


def _default_preprocess_hooks() -> list[str]:
    """Full preprocess hook list (ignore transient ``GOPEX_AV_FOLD_HOOKS`` for product stacks)."""
    from ltx_trainer.fold_registry import fold_hooks_for_preprocess_meta

    saved = os.environ.pop("GOPEX_AV_FOLD_HOOKS", None)
    try:
        return list(fold_hooks_for_preprocess_meta())
    finally:
        if saved is not None:
            os.environ["GOPEX_AV_FOLD_HOOKS"] = saved


def curated_innovations() -> list[Innovation]:
    return list(_CURATED)


def innovations_by_pillar() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {p.value: [] for p in Pillar}
    for inv in _CURATED:
        out[inv.pillar.value].append(_innovation_dict(inv))
    return out


def _innovation_dict(inv: Innovation) -> dict[str, Any]:
    d = asdict(inv)
    d["pillar"] = inv.pillar.value
    d["tier"] = inv.tier.value
    return d


def paper_coverage() -> dict[str, Any]:
    """Compare fold registry vs curated LTX 3 stack."""
    registered = set(list_registered_hooks())
    curated_hooks = {inv.hook for inv in _CURATED if inv.hook}
    stack_hooks: set[str] = set()
    for stack in STACKS.values():
        stack_hooks.update(stack.hooks)
    full_stack = STACKS["ltx3_full"]
    full_hooks = _default_preprocess_hooks() if not full_stack.hooks else full_stack.hooks

    missing_from_registry = sorted(curated_hooks - registered)
    registry_only = sorted(registered - curated_hooks - set(full_hooks))

    return {
        "registered_hook_count": len(registered),
        "curated_innovation_count": len(_CURATED),
        "curated_with_hooks": len(curated_hooks),
        "default_preprocess_hook_count": len(full_hooks),
        "tiers": _tier_counts(),
        "pillars": {k: len(v) for k, v in innovations_by_pillar().items()},
        "missing_curated_hooks_in_registry": missing_from_registry,
        "registry_hooks_not_in_curated_sample": registry_only[:40],
        "registry_hooks_not_in_curated_total": len(registry_only),
        "acceptance_gates": {
            "G1": [i.id for i in _CURATED if i.gate == "G1"],
            "G2": [i.id for i in _CURATED if i.gate == "G2"],
            "G3": [i.id for i in _CURATED if i.gate == "G3"],
            "G4": [i.id for i in _CURATED if i.gate == "G4"],
        },
    }


def _tier_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for inv in _CURATED:
        counts[inv.tier.value] = counts.get(inv.tier.value, 0) + 1
    return counts


def innovation_stack(name: str = "ltx3_full") -> dict[str, Any]:
    """Resolve a named LTX 3 stack to hooks, env, and recommended commands."""
    if name not in STACKS:
        known = ", ".join(sorted(STACKS))
        raise KeyError(f"unknown stack {name!r} (try: {known})")
    stack = STACKS[name]
    hooks = list(stack.hooks)
    if name == "ltx3_full" and not hooks:
        hooks = _default_preprocess_hooks()
    inv_ids = {i.id for i in _CURATED}
    related = [_innovation_dict(i) for i in _CURATED if i.hook in hooks or i.id in inv_ids]
    return {
        "stack": stack.name,
        "description": stack.description,
        "pillars": [p.value for p in stack.pillars],
        "hooks": hooks,
        "env": dict(stack.env),
        "train_profile": stack.train_profile,
        "gates": list(stack.gates),
        "commands": list(stack.commands),
        "GOPEX_AV_FOLD_HOOKS": ",".join(hooks),
        "curated_matches": related[:24],
    }


def ltx3_train_hooks() -> list[str]:
    """Hooks for ``ltx3_quality`` research profile (train gates + minute/HDR routing)."""
    st = innovation_stack("ltx3_full")
    # Prefer quality-loop + minute pillars; skip pure panorama unless env requests it.
    core = innovation_stack("ltx3_minute")["hooks"] + innovation_stack("ltx3_hdr")["hooks"]
    merged: list[str] = []
    seen: set[str] = set()
    for h in core + st["hooks"]:
        if h not in seen:
            seen.add(h)
            merged.append(h)
    return merged


def evolve_innovation_summary() -> dict[str, Any]:
    """Compact card for ``kino-ltx3-evolve.sh innovations``."""
    cov = paper_coverage()
    return {
        "vision": "documents/LTX_3_VISION.md",
        "pillars": [p.value for p in Pillar],
        "stacks": sorted(STACKS.keys()),
        "coverage": cov,
        "deep_integrations": [
            i.id
            for i in _CURATED
            if i.tier in (Tier.INFER, Tier.COMPOSE, Tier.PRODUCT, Tier.TRAIN_GATE)
        ],
    }
