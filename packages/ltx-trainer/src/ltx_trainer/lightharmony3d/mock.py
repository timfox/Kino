"""Runnable evaluation smoke for LightHarmony3D (arXiv:2603.29209)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lightharmony3d.benchmarks import PAPER_ARXIV, TABLE1_LH3D_KU, benchmarks_bundle


def evaluation_smoke() -> dict[str, Any]:
    ours = TABLE1_LH3D_KU["lightharmony3d"]
    gas = TABLE1_LH3D_KU["gaslight"]
    out: dict[str, Any] = {
        "package": "lightharmony3d",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_lh3d_ku_psnr": ours["psnr"],
        "beats_gaslight_psnr": ours["psnr"] > gas["psnr"],
    }
    try:
        import torch

        from ltx_trainer.lightharmony3d.config import LightHarmony3DConfig
        from ltx_trainer.lightharmony3d.hdr_fusion import fuse_exposure_bracket
        from ltx_trainer.lightharmony3d.model import LightHarmony3D
        from ltx_trainer.lightharmony3d.pipeline import train_step
        from ltx_trainer.lightharmony3d.synthetic import synthesize_insertion_pair, synthesize_scene

        model = LightHarmony3D(LightHarmony3DConfig(image_size=32))
        n_params = sum(p.numel() for p in model.parameters())

        scene, obj, mask, gt = synthesize_insertion_pair(32, seed=0)
        with torch.no_grad():
            result = model(scene.unsqueeze(0), obj.unsqueeze(0), mask.unsqueeze(0))

        ev0 = synthesize_scene(32, seed=1).unsqueeze(0)
        dark = (ev0 * 0.12).clamp(0, 1)
        hdr = fuse_exposure_bracket([dark, ev0 * 0.25, ev0], [-6, -3, 0])

        model.train()
        loss, stats = train_step(
            model,
            background=scene.unsqueeze(0),
            object_rgb=obj.unsqueeze(0),
            mask=mask.unsqueeze(0),
            target=gt.unsqueeze(0),
        )

        out.update(
            {
                "torch": True,
                "params_k": round(n_params / 1000.0, 2),
                "composite_mean": round(float(result.composite.mean()), 4),
                "hdr_max": round(float(hdr.max()), 4),
                "loss_l1": round(float(loss.item()), 5),
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
