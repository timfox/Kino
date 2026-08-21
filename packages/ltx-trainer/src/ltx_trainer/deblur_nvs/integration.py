"""LTX / GOPEX integration notes for blur-aware NVS."""

from __future__ import annotations

from typing import Any


def gopex_links() -> dict[str, Any]:
    return {
        "gphotos_video": {
            "script": "./scripts/kino-gphotos-native-train.sh",
            "role": "Handheld phone clips often carry motion blur; sparse sharp views help I2V conditioning",
        },
        "control_room_17": {
            "agent_doc": "data/control_room_17/AGENTS.md",
            "role": "Reshoot mismatches when blur breaks cross-view geometry in sparse teleplay refs",
        },
        "lucky_hdr": {
            "doc": "documents/LUCKY_HDR.md",
            "role": "Handheld bracket alignment shares blur/exposure degradation themes",
        },
        "erpgs": {
            "doc": "documents/ERPGS.md",
            "role": "ERP 3DGS NVS baseline; DeblurNVS targets feed-forward blur-aware alternative",
        },
        "face_age_10k": {
            "agent_doc": "data/face_age_10k/AGENTS.md",
            "role": "Age-band QA on cast refs after blur-aware restoration sharpens facial detail",
        },
        "face_rec_survey": {
            "doc": "documents/FACE_REC_SURVEY.md",
            "role": "Pose/occlusion/illumination confounders for CR17 cast_refs anchor QA",
        },
        "dl3dv": {
            "hub": "DL3DV-10K",
            "role": "Upstream sharp multi-view source for DL3DV-10K-Blur training pairs",
        },
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Prefer 3+ context views at similar exposure when blur is moderate; DeblurNVS paper uses K=3 default.",
        "Motion-blurred gphotos prep: run parallel prep with GOPEX_AUDIT_RELAX_STRICT=1 if latents fail strict audit.",
        "Geometry-aware latent restoration avoids per-view 2D deblur artifacts before LTX I2V encode.",
        "Pairs with proceduralsky / HDR ERP: blur-aware NVS restores sharp context before pano-native training.",
    ]


def ltx_plan_stub() -> dict[str, Any]:
    """CPU plan card — no upstream DeblurNVS weights bundled in GOPEX."""
    return {
        "phase": "blur_aware_sparse_nvs",
        "upstream": "https://github.com/PKU-YuanGroup/DeblurNVS",
        "inputs": ["motion_blurred_context_views", "target_camera_pose"],
        "outputs": ["sharp_novel_view_rgb", "restored_context_latents"],
        "gopex_stub": "ltx_trainer.deblur_nvs.pipeline.infer_novel_view",
        "fold_hook": None,
        "notes": ltx_prep_notes(),
    }
