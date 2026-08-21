"""GOPEX integration: PAPR rig animation + I2V driving video."""

from __future__ import annotations

from typing import Any


def gopex_stub_links() -> dict[str, str]:
    return {
        "diffusion_lm": "I2V fixed-viewpoint driving (Kling / LTX ti2vid)",
        "openvid_1m": "T2V/I2V prompt corpus for driving sequences",
        "vidprom": "Prompt gallery + NSFW scores for I2V planning",
        "rapidata_i2v_pref": "Human preference on I2V outputs",
        "erpgs": "ERP Gaussian splats (contrast: per-primitive covariance under LBS)",
        "r5dgs": "Semantic 4DGS + rigid constraints",
        "p2gs": "HDR exposure-invariant 3DGS",
        "control_room_17": "Teleplay I2V shot ladder for character motion refs",
        "face_age_10k": "Character age consistency on animated subjects",
    }


def ltx_plan_stub() -> dict[str, Any]:
    return {
        "phase": "fixed_viewpoint_rig_animation",
        "pipeline": [
            "multiview_papr_pretrain_canonical",
            "fixed_view_i2v_driving_video",
            "offsetopt_mesh_proxy_puppeteer_autorig",
            "phase1_depth_supervised_joint_rotations",
            "phase2_track_arap_skinning_refine",
            "novel_view_papr_render",
        ],
        "representation": "PAPR (interpolation-based, no per-primitive shape)",
        "deformation": "direct LBS on point positions; features rigid",
        "supervision": "single fixed camera + decaying depth/track priors",
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Pretrain static PAPR from ring cameras (Objaverse synthetic or real MV capture).",
        "Generate or capture fixed-viewpoint driving video (512² synth / 960×540 real).",
        "Auto-rig via OffsetOPT mesh → Puppeteer; discard proxy after IDW weight transfer.",
        "Phase 1: sequential per-frame joint rotation + DA3 relative depth (no tracks).",
        "Phase 2: joint-level correction MLP + skinning logits; CoTracker from attention-dominant seeds.",
        "Inference: single forward LBS + frozen PAPR renderer at novel views / novel poses.",
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_stubs": gopex_stub_links(),
        "ltx_plan": ltx_plan_stub(),
        "prep_notes": ltx_prep_notes(),
    }
