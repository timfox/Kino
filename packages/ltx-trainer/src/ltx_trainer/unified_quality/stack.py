"""Unified research stack: train + infer roles for video quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Phase = Literal["train", "infer", "both"]


@dataclass(frozen=True)
class StackLayer:
    id: str
    paper: str
    phase: Phase
    role: str
    env: tuple[str, ...] = ()
    script: str = ""


UNIFIED_QUALITY_STACK: tuple[StackLayer, ...] = (
    StackLayer(
        "avbench",
        "AVBench",
        "train",
        "Sample weighting from human-aligned T2AV metrics in AV-fold sidecars",
    ),
    StackLayer(
        "mtavg2",
        "MTAVG-Bench 2.0",
        "train",
        "Downweight cinematic expressiveness failure risk during connector evolve",
        "use_mtavg2_expressiveness_weights",
    ),
    StackLayer(
        "growloop",
        "GrowLoop",
        "train",
        "Human-likeness / anti-uncanny sample weights in AV-fold",
        "use_growloop_weights",
    ),
    StackLayer(
        "adamag",
        "AdaMaG",
        "infer",
        "Probability-conserving time-varying CFG in HQ stage 1",
        "GOPEX_INFER_ADAMAG",
        "pipeline/two_stage_hq_kino.py",
    ),
    StackLayer(
        "its_avgen",
        "ITS-AVGen",
        "infer",
        "Best-of-N candidate ranking at delivery",
        "GOPEX_ITS_ENABLED,GOPEX_ITS_N_CANDIDATES",
        "tools/its_rank_candidates.py",
    ),
    StackLayer(
        "hdr_lift",
        "SDR2HDR / tonemap",
        "infer",
        "Post-render HDR lift on delivery batch",
        "GOPEX_DELIVERY_HDR_LIFT,GOPEX_SDR2HDR_LIFT_MODE",
    ),
    StackLayer(
        "lalqr",
        "LA-LQR",
        "infer",
        "Reduced-order LQR safety steering on text cross-attn context",
        "GOPEX_INFER_LALQR,GOPEX_LALQR_CATEGORY",
        "kino/packages/ltx-trainer/src/ltx_trainer/lalqr/infer_bridge.py",
    ),
    StackLayer(
        "quality_max",
        "quality_max profile",
        "both",
        "Stricter fold floors + orchestrated train/infer defaults",
        "GOPEX_RESEARCH_PROFILE,GOPEX_INFER_QUALITY_TIER",
        "scripts/kino-unified-quality.sh",
    ),
)


def stack_card(profile: str = "quality_max") -> dict[str, Any]:
    """Human-readable map of how papers combine for a profile."""
    train_layers = [layer for layer in UNIFIED_QUALITY_STACK if layer.phase in ("train", "both")]
    infer_layers = [layer for layer in UNIFIED_QUALITY_STACK if layer.phase in ("infer", "both")]
    return {
        "profile": profile,
        "train": [{"id": l.id, "paper": l.paper, "role": l.role} for l in train_layers],
        "infer": [{"id": l.id, "paper": l.paper, "role": l.role, "env": list(l.env)} for l in infer_layers],
        "pipeline": [
            "profile + YAML patch",
            "AV-fold backfill (mtavg2, growloop, avbench, …)",
            "connector evolve @ quality_max gates",
            "pre-delivery quality gate",
            "HQ delivery: AdaMaG + LA-LQR safety + ITS Best-of-N + HDR lift",
        ],
    }


def verify_stack(repo_root: str) -> dict[str, Any]:
    """Check key scripts/modules exist for the unified stack."""
    from pathlib import Path

    root = Path(repo_root)
    checks: list[dict[str, Any]] = []

    def _check(name: str, path: str) -> None:
        p = root / path
        checks.append({"name": name, "path": path, "ok": p.exists()})

    _check("hq_adamag", "kino/packages/ltx-trainer/src/ltx_trainer/adamag/hq_guidance.py")
    _check("hq_pipeline", "pipeline/two_stage_hq_kino.py")
    _check("its_ranker", "tools/its_rank_candidates.py")
    _check("video_enhance", "scripts/kino-video-enhance.sh")
    _check("infer_quality", "scripts/lib/kino_infer_quality.sh")
    _check("lalqr_bridge", "kino/packages/ltx-trainer/src/ltx_trainer/lalqr/infer_bridge.py")
    _check("quality_gate", "scripts/kino-ltx-quality-gate.sh")
    _check("research_profile", "tools/ltx_research_profile.py")
    _check("evolve_yaml", "kino/packages/ltx-trainer/configs/ltx2_av_lora_gemma4_31b_native_connectors_evolve.yaml")

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks, "profile_default": "quality_max"}
