"""PIU pipeline glue and reference tables (Sec. 4)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.piu.anchor import build_centroids_from_clusters, select_anchor_identity
from ltx_trainer.piu.conditioning import sample_forget_batch
from ltx_trainer.piu.config import BASELINES, PIUConfig
from ltx_trainer.piu.layers import SURGICAL_BLOCKS, SURGICAL_PARAM_FRACTION
from ltx_trainer.piu.losses import total_piu_loss


class DemoNoisePredictor(nn.Module):
    """Tiny stand-in for ε_θ(z,t,c) smoke tests (not Arc2Face)."""

    def __init__(self, dim: int = 64, cond_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z: Tensor, t: Tensor, c: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.view(1)
        t_feat = t.float().view(-1, 1) / 1000.0
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        if t_feat.shape[0] == 1 and z.shape[0] > 1:
            t_feat = t_feat.expand(z.shape[0], -1)
        x = torch.cat([z, c, t_feat], dim=-1)
        return self.net(x)


def training_step(
    model: nn.Module,
    frozen: nn.Module,
    forget_pool: Tensor,
    retain_pool: Tensor,
    anchor_cond: Tensor,
    *,
    cfg: PIUConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    """One PIU optimization step with Gaussian z_t (Sec. 3.4)."""
    cfg = cfg or PIUConfig()
    device = device or torch.device("cpu")
    b = min(cfg.batch_size, forget_pool.shape[0], retain_pool.shape[0])
    dim = 64
    z = torch.randn(b, dim, device=device)
    t = torch.randint(0, 1000, (b,), device=device)
    c_f = sample_forget_batch(forget_pool.to(device), b, alpha=cfg.dirichlet_alpha)
    c_r = retain_pool[torch.randint(0, retain_pool.shape[0], (b,))].to(device)
    c_a = anchor_cond.to(device).unsqueeze(0).expand(b, -1)

    eps_tf = model(z, t, c_f)
    eps_tr = model(z, t, c_r)
    with torch.no_grad():
        eps_ff = frozen(z, t, c_f)
        eps_fa = frozen(z, t, c_a)
        eps_fr = frozen(z, t, c_r)

    losses = total_piu_loss(eps_tf, eps_tr, eps_ff, eps_fa, eps_fr, cfg=cfg)
    return {k: float(v.item()) for k, v in losses.items()}


def benchmark_table_sota(cfg: PIUConfig | None = None) -> dict[str, dict[str, float]]:
    """Table 1 comparison on CelebA-HQ / Arc2Face."""
    cfg = cfg or PIUConfig()
    return {
        "arc2face": {
            "forget_ism": cfg.baseline_forget_ism,
            "retain_ism": cfg.baseline_retain_ism,
            "srk": 0.99,
            "delta_kd_forget": 0.0,
            "delta_kd_retain": 0.0,
            "ediffiqa_forget": 0.76,
            "ediffiqa_retain": 0.76,
        },
        "siss": {
            "forget_ism": 0.59,
            "retain_ism": 0.58,
            "srk": 0.98,
            "delta_kd_forget": 5.23,
            "delta_kd_retain": 8.14,
            "ediffiqa_forget": 0.65,
            "ediffiqa_retain": 0.64,
        },
        "uce": {
            "forget_ism": 0.59,
            "retain_ism": 0.70,
            "srk": 1.00,
            "delta_kd_forget": 3.69,
            "delta_kd_retain": 1.26,
            "ediffiqa_forget": 0.75,
            "ediffiqa_retain": 0.75,
        },
        "wid": {
            "forget_ism": 0.32,
            "retain_ism": 0.44,
            "srk": 49.28,
            "delta_kd_forget": 4.35,
            "delta_kd_retain": 3.36,
            "ediffiqa_forget": 0.76,
            "ediffiqa_retain": 0.75,
        },
        "piu": {
            "forget_ism": cfg.piu_forget_ism,
            "retain_ism": cfg.piu_retain_ism,
            "srk": cfg.piu_srk,
            "delta_kd_forget": 8.12,
            "delta_kd_retain": 0.39,
            "ediffiqa_forget": 0.75,
            "ediffiqa_retain": 0.75,
        },
    }


def ablation_table_components() -> dict[str, dict[str, float]]:
    """Table 2 incremental PIU components."""
    return {
        "baseline": {"forget_ism": 0.760, "retain_ism": 0.734, "srk": 0.99},
        "naive": {"forget_ism": 0.250, "retain_ism": 0.410, "srk": 55.6},
        "plus_preservation": {"forget_ism": 0.564, "retain_ism": 0.716, "srk": 1.6},
        "plus_neg_guidance": {"forget_ism": 0.298, "retain_ism": 0.714, "srk": 74.1},
        "plus_surgical": {"forget_ism": 0.280, "retain_ism": 0.709, "srk": 90.1},
    }


def ablation_lambda_table() -> dict[str, dict[str, float]]:
    """Table 3 preservation weight λ."""
    return {
        "0": {"forget_ism": 0.048, "retain_ism": 0.229, "srk": 11.2},
        "5": {"forget_ism": 0.213, "retain_ism": 0.700, "srk": 90.3},
        "10": {"forget_ism": 0.280, "retain_ism": 0.709, "srk": 90.1},
        "20": {"forget_ism": 0.376, "retain_ism": 0.712, "srk": 27.2},
    }


def ablation_tau_table() -> dict[str, dict[str, float]]:
    """Table 4 anchor proximity τ."""
    return {
        "0.1": {"forget_ism": 0.289, "retain_ism": 0.712, "srk": 65.82},
        "0.2": {"forget_ism": 0.392, "retain_ism": 0.713, "srk": 46.57},
        "0.3": {"forget_ism": 0.463, "retain_ism": 0.716, "srk": 23.92},
        "0.4": {"forget_ism": 0.449, "retain_ism": 0.714, "srk": 42.15},
        "0.5": {"forget_ism": 0.451, "retain_ism": 0.713, "srk": 42.68},
    }


def ablation_eta_table() -> dict[str, dict[str, float]]:
    """Table 5 negative guidance η."""
    return {
        "0.0": {"forget_ism": 0.564, "retain_ism": 0.716, "srk": 1.6},
        "1.0": {"forget_ism": 0.341, "retain_ism": 0.721, "srk": 82.1},
        "1.5": {"forget_ism": 0.283, "retain_ism": 0.720, "srk": 90.1},
        "3.0": {"forget_ism": 0.185, "retain_ism": 0.694, "srk": 90.4},
    }


def ablation_layers_table() -> dict[str, dict[str, float]]:
    """Table 6 fine-tuning scope."""
    return {
        "full": {"forget_ism": 0.134, "retain_ism": 0.294, "srk": 51.9, "params": "900M"},
        "full_ca": {"forget_ism": 0.298, "retain_ism": 0.714, "srk": 74.1, "params": "44M"},
        "surgical": {"forget_ism": 0.280, "retain_ism": 0.709, "srk": 90.1, "params": "37M"},
    }


def demo_identity_clusters(
    *,
    num_identities: int = 20,
    samples_per_id: int = 8,
    dim: int = 512,
    seed: int = 0,
) -> dict[int, Tensor]:
    """Synthetic ArcFace-like clusters for anchor selection demos."""
    g = torch.Generator().manual_seed(seed)
    clusters: dict[int, Tensor] = {}
    for i in range(num_identities):
        center = torch.randn(dim, generator=g)
        noise = 0.1 * torch.randn(samples_per_id, dim, generator=g)
        clusters[i] = center.unsqueeze(0) + noise
    return clusters


def demo_anchor_selection(
    forget_id: int = 0,
    clusters: dict[int, Tensor] | None = None,
    *,
    cfg: PIUConfig | None = None,
) -> dict[str, Any]:
    clusters = clusters or demo_identity_clusters()
    centroids = build_centroids_from_clusters(clusters)
    mu_f = centroids[forget_id]
    others = {k: v for k, v in centroids.items() if k != forget_id}
    anchor_id = select_anchor_identity(mu_f, others, cfg=cfg, forget_id=forget_id)
    from ltx_trainer.piu.anchor import cosine_similarity

    return {
        "forget_id": forget_id,
        "anchor_id": anchor_id,
        "cosine_to_anchor": cosine_similarity(mu_f, centroids[anchor_id]),
        "tau": (cfg or PIUConfig()).anchor_tau,
    }


def dataset_card(cfg: PIUConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PIUConfig()
    return {
        "name": "PIU",
        "paper": "arXiv:2605.22311",
        "title": "Proximity-guided Identity Unlearning in ID-Conditioned Diffusion Models",
        "base_model": "Arc2Face",
        "dataset": "CelebA-HQ",
        "identity_clusters": 9683,
        "preservation_lambda": cfg.preservation_lambda,
        "anchor_tau": cfg.anchor_tau,
        "negative_guidance_eta": cfg.negative_guidance_eta,
        "surgical_blocks": list(SURGICAL_BLOCKS),
        "surgical_param_fraction": SURGICAL_PARAM_FRACTION,
        "baselines": list(BASELINES),
        "code": "https://github.com/edgarcancinoe/piu_unlearning",
    }
