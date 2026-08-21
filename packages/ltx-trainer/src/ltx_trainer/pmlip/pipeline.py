"""Training, inference, checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.pmlip.egnn import PEGNN
from ltx_trainer.pmlip.losses import PMLIPLoss
from ltx_trainer.pmlip.metrics import spread_to_skill_ratio, uncertainty_spearman
from ltx_trainer.pmlip.model import PMLIPConfig, PerturbedMLIP


def train_step(
    model: PEGNN | PerturbedMLIP,
    loss_fn: PMLIPLoss,
    *,
    h: Tensor,
    x: Tensor,
    edge_index: Tensor,
    target: Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(model, PerturbedMLIP):
        model = model.model  # unwrap to PEGNN
    samples = model.predict(h, x, edge_index, k=loss_fn.cfg.k_train)
    return loss_fn(samples, target)


@torch.no_grad()
def predict_with_uncertainty(
    model: PEGNN | PerturbedMLIP,
    h: Tensor,
    x: Tensor,
    edge_index: Tensor,
    *,
    k: int = 50,
) -> dict[str, Tensor | float]:
    if isinstance(model, PerturbedMLIP):
        pegnn = model.model
    else:
        pegnn = model
    samples = pegnn.predict(h, x, edge_index, k=k)
    mean = samples.mean(0)
    var = samples.var(0, unbiased=False)
    return {"mean": mean, "samples": samples, "variance": var}


def load_pmlip_checkpoint(path: Path | str, *, device: str = "cpu", backend: str = "egnn") -> PerturbedMLIP:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = PMLIPConfig(**ckpt.get("config", {}))
    model = PerturbedMLIP(cfg, backend=backend)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()


def save_checkpoint(model: PerturbedMLIP | PEGNN, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, PEGNN):
        wrapper = PerturbedMLIP(model.cfg if hasattr(model, "cfg") else PMLIPConfig())
        wrapper.model = model
        state = wrapper.state_dict()
        cfg = PMLIPConfig()
    else:
        state = model.state_dict()
        cfg = model.cfg
    torch.save({"config": asdict(cfg), "state_dict": state}, path)


def evaluate_batch(
    model: PEGNN,
    h: Tensor,
    x: Tensor,
    edge_index: Tensor,
    target: Tensor,
    *,
    k: int = 100,
) -> dict[str, float]:
    samples = model.predict(h, x, edge_index, k=k)
    mean = samples.mean(0)
    mse = float(((mean - target) ** 2).mean().detach())
    from ltx_trainer.pmlip.crps import fair_crps_multivariate

    crps = float(fair_crps_multivariate(samples.reshape(k, -1), target.reshape(-1)))
    ssr = spread_to_skill_ratio(samples.reshape(k, -1), target.reshape(-1), k=k)
    err = ((mean - target) ** 2).sum(dim=-1)
    var = samples.var(0, unbiased=False).sum(dim=-1)
    spear = uncertainty_spearman(err, var)
    return {"mse": mse, "crps": crps, "ssr": ssr, "spearman": spear}
