"""EVIDENT framework card and reference experiment tables."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.evident.config import EVIDENTConfig
from ltx_trainer.evident.e2v_gating import e2v_gating_scores
from ltx_trainer.evident.eb_adapter import EntityBottleneckAdapter
from ltx_trainer.evident.eb_distillation import entity_binding_distillation_loss, kmeans_cluster_maps


def framework_card(cfg: EVIDENTConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EVIDENTConfig()
    return {
        "name": "EVIDENT",
        "paper": "arXiv:2605.26104",
        "title": "Routing MLLM Adaptation through Entity-Grounded Visual Evidence for Cross-Domain VTG",
        "backbone": cfg.backbone,
        "components": {
            "eb_adapter": f"K={cfg.num_slots} slots, layers {list(cfg.adapter_layers)[:3]}…",
            "eb_distillation": f"DINOv2 K-means + Hungarian BCE (λ={cfg.eb_distill_weight})",
            "e2v_gating": "subject×object co-occurrence per frame",
        },
        "trainable_params_m": cfg.trainable_params_m,
        "finding": "OOD VTG failure driven by visual domain shift, not unseen query concepts",
    }


def table_cross_domain_vbench() -> dict[str, dict[str, dict[str, float]]]:
    """Table 1 — R1@0.5 / R1@0.7 highlights (Cha.→QVH and QVH→Cha.)."""
    return {
        "charades_sta_source": {
            "qwen_lora": {"id_r1_05": 67.7, "id_r1_07": 43.9, "ood_r1_05": 54.3, "ood_r1_07": 29.7},
            "evident": {"id_r1_05": 69.1, "id_r1_07": 49.2, "ood_r1_05": 59.8, "ood_r1_07": 36.9},
        },
        "qvhighlights_source": {
            "qwen_lora": {"id_r1_05": 78.2, "id_r1_07": 64.8, "ood_r1_05": 48.1, "ood_r1_07": 26.2},
            "evident": {"id_r1_05": 82.2, "id_r1_07": 69.1, "ood_r1_05": 50.2, "ood_r1_07": 28.0},
        },
    }


def table_ablation_components() -> dict[str, dict[str, float]]:
    """Table 2(a) — Cha.→QVH component ablation."""
    return {
        "lora_baseline": {"id_r1_05": 67.7, "id_r1_07": 43.9, "ood_r1_05": 54.3, "ood_r1_07": 29.7},
        "eb_adapter_only": {"id_r1_05": 68.1, "id_r1_07": 49.5, "ood_r1_05": 54.9, "ood_r1_07": 33.1},
        "eb_adapter_ebd": {"id_r1_05": 69.1, "id_r1_07": 50.4, "ood_r1_05": 58.8, "ood_r1_07": 36.7},
        "evident_full": {"id_r1_05": 69.1, "id_r1_07": 49.2, "ood_r1_05": 59.8, "ood_r1_07": 36.9},
    }


def table_ablation_adapter_design() -> dict[str, dict[str, float]]:
    """Table 2(b) — adapter placement."""
    return {
        "after_vision_encoder": {"id_r1_05": 34.6, "ood_r1_05": 19.5},
        "adapter_style": {"id_r1_05": 69.1, "ood_r1_05": 59.8},
    }


def table_domain_gap_analysis() -> dict[str, float]:
    """Figure 3 — concept vs visual gap on Cha.→QVH."""
    return {
        "seen_vs_unseen_concepts_r1_gap": 3.7,
        "visually_similar_r1_05": 52.8,
        "visually_dissimilar_r1_05": 39.1,
        "visual_gap_points": 13.7,
        "gt_perturb_drop_id": 17.4,
        "random_perturb_drop_id": 9.6,
        "gt_perturb_drop_ood": 12.6,
        "random_perturb_drop_ood": 12.1,
    }


def training_step_demo(
    cfg: EVIDENTConfig | None = None,
    *,
    batch_size: int = 2,
    device: str = "cpu",
) -> dict[str, float]:
    """Smoke: EB Adapter + EBD + E2V on synthetic tokens."""
    cfg = cfg or EVIDENTConfig()
    dev = torch.device(device)
    b = batch_size
    t, n = 2, cfg.tokens_per_frame
    text_len = 8
    seq_len = t * n + text_len
    x = torch.randn(b, seq_len, cfg.hidden_dim, device=dev)

    adapter = EntityBottleneckAdapter(cfg).to(dev)
    frame_feats = torch.randn(b, t, cfg.bottleneck_dim, device=dev)
    g = e2v_gating_scores(
        frame_feats[0],
        torch.randn(cfg.bottleneck_dim, device=dev),
        torch.randn(cfg.bottleneck_dim, device=dev),
    )
    x_out, attn = adapter(x, text_len=text_len, gating=g.unsqueeze(0).expand(b, -1))

    pseudo = torch.randn(t, n, 32, device=dev)
    clusters = kmeans_cluster_maps(pseudo, cfg.num_slots, steps=3)
    attn_kn = attn[0].transpose(0, 1).unsqueeze(0).expand(t, -1, -1)
    loss_ebd = float(entity_binding_distillation_loss(attn_kn, clusters).item())

    return {
        "output_delta_norm": float((x_out - x).norm().item()),
        "ebd_loss": loss_ebd,
        "mean_gating": float(g.mean()),
        "num_slots": float(cfg.num_slots),
    }


def evaluation_demo(*, device: str = "cpu") -> dict[str, Any]:
    step = training_step_demo(device=device)
    tab = table_cross_domain_vbench()
    return {
        **step,
        "evident_ood_r1_05_cha_to_qvh": tab["charades_sta_source"]["evident"]["ood_r1_05"],
        "lora_ood_r1_05_cha_to_qvh": tab["charades_sta_source"]["qwen_lora"]["ood_r1_05"],
        "ood_gain_r1_05": (
            tab["charades_sta_source"]["evident"]["ood_r1_05"]
            - tab["charades_sta_source"]["qwen_lora"]["ood_r1_05"]
        ),
        "visual_gap_points": table_domain_gap_analysis()["visual_gap_points"],
    }
