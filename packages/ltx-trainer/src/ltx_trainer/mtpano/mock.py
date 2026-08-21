"""Evaluation smoke (arXiv:2602.05330)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mtpano.benchmarks import TABLE1_STRUCTURED3D, benchmarks_bundle
from ltx_trainer.mtpano.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    ours = next(r for r in TABLE1_STRUCTURED3D if r["method"] == "Ours")
    out: dict[str, Any] = {
        "package": "mtpano",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_mIoU": ours["mIoU"],
    }
    try:
        import torch

        from ltx_trainer.mtpano.auxiliary import build_auxiliary_targets
        from ltx_trainer.mtpano.config import MTPanoConfig
        from ltx_trainer.mtpano.erp import ERPTokenMixer, latitude_weight
        from ltx_trainer.mtpano.patch_pipeline import (
            e2p_normals,
            p2e_patch_stub,
            sample_perspective_pose,
            yaw_pitch_rotation_matrix,
        )
        from ltx_trainer.mtpano.pd_bridgenet import PDBridgeNetStub
        from ltx_trainer.mtpano.pipeline import evaluation_demo_run

        cfg = MTPanoConfig(height=64, width=128)
        demo = evaluation_demo_run(cfg, device="cpu")
        model = PDBridgeNetStub(cfg)
        img = torch.rand(1, 3, 64, 128)
        preds = model(img)
        yaw, pitch, _ = sample_perspective_pose(1)
        patch = p2e_patch_stub(img, yaw, pitch)
        w = float(latitude_weight(torch.tensor(0.1)).item())
        phi = torch.zeros(16, 32)
        mixer = ERPTokenMixer(8)
        tok = mixer(torch.randn(1, 8, 16, 32), phi)
        aux = build_auxiliary_targets(img, preds["depth"])
        rot = yaw_pitch_rotation_matrix(yaw, pitch)
        n_sphere = e2p_normals(torch.randn(1, 3, 16, 32), rot)
        out.update(
            {
                "torch": True,
                "demo_loss": demo["train"]["loss"],
                "semseg_shape": list(preds["semseg"].shape),
                "patch_shape": list(patch.shape),
                "mixer_shape": list(tok.shape),
                "latitude_w": w,
                "edf_shape": list(aux["edf"].shape),
                "point_map_shape": list(aux["point_map"].shape),
                "normal_e2p_shape": list(n_sphere.shape),
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
