"""Training smoke and demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.anything360.anything360_model import Anything360Stub
from ltx_trainer.anything360.benchmarks import table1_laval
from ltx_trainer.anything360.circular_latent import StubVAEEncoder, discontinuity_score
from ltx_trainer.anything360.config import Anything360Config
from ltx_trainer.anything360.synthetic import synthetic_pair


def train_step(cfg: Anything360Config | None = None) -> dict[str, float]:
    cfg = cfg or Anything360Config()
    model = Anything360Stub(cfg)
    model.train()
    pers, erp = synthetic_pair(cfg, batch_size=1)
    out = model(pers, erp)
    loss = torch.nn.functional.mse_loss(out["z_pred"], out["noise_target"])
    loss.backward()
    return {"loss": float(loss.detach())}


def cle_ablation_demo(cfg: Anything360Config | None = None) -> dict[str, float]:
    cfg = cfg or Anything360Config(erp_height=64, erp_width=128)
    erp = torch.rand(1, 3, cfg.erp_height, cfg.erp_width)
    enc = StubVAEEncoder(3, cfg.latent_channels)
    z_vanilla = enc(erp, circular=False)
    z_cle = enc(erp, circular=True)
    return {"DS_vanilla": discontinuity_score(z_vanilla), "DS_CLE": discontinuity_score(z_cle)}


def evaluation_demo_run(cfg: Anything360Config | None = None) -> dict[str, Any]:
    cfg = cfg or Anything360Config()
    model = Anything360Stub(cfg)
    model.eval()
    pers, erp = synthetic_pair(cfg, batch_size=1)
    with torch.no_grad():
        out = model(pers, erp)
    cle = cle_ablation_demo(cfg)
    return {
        "erp_recon_shape": list(out["erp_recon"].shape),
        "sequence_concat": cfg.sequence_concat,
        "use_circular_latent": cfg.use_circular_latent,
        "cle_DS": cle,
        "reference_laval_FID": table1_laval()["FID"],
    }
