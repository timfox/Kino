"""EchoPilot framework card, benchmark tables, and smoke demos."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.echopilot.config import EchoPilotConfig
from ltx_trainer.echopilot.memory import (
    feature_consistency,
    gate_memory_sequence,
    masked_descriptor,
    reliability_write_gate,
)
from ltx_trainer.echopilot.prompting import (
    crop_boxes,
    random_vfm_features,
    stub_attribution_from_similarity,
    synthesize_prompts,
    vfm_cosine_map,
)
from ltx_trainer.echopilot.seed import (
    select_scale_seed,
    semantic_similarity,
    spatial_seed_score,
)


def framework_card(cfg: EchoPilotConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EchoPilotConfig()
    return {
        "name": "EchoPilot",
        "paper": "arXiv:2605.25944",
        "title": "Training-Free Ultrasound Video Segmentation via Scale-Space Semantic Prompting",
        "setting": "single first-frame point + anatomical category name; no finetuning",
        "stage_i": "S.E.E.D. scale selection + VFM auxiliary prompts",
        "stage_ii": "reliability-gated memory (τ={})".format(cfg.memory_gate_tau),
        "vlm": cfg.vlm_backbone,
        "vfm": cfg.vfm_backbone,
        "segmentors": list(cfg.segmentor_backends),
        "datasets": ["CAMUS", "Breast Lesion", "Placenta"],
        "placenta_frames": cfg.placenta_frames,
        "project_page": cfg.project_page,
    }


def table_main_results() -> dict[str, dict[str, dict[str, float]]]:
    """Table 1 — Dice (D), ASD (A), F-boundary (F) on three datasets."""
    return {
        "SAM2": {
            "sam2": {"D": 28.41, "A": 56.84, "F": 0.80},
            "ma_sam2": {"D": 28.43, "A": 56.81, "F": 0.81},
            "sam2long": {"D": 28.44, "A": 56.65, "F": 0.80},
            "echopilot": {"D": 34.09, "A": 28.86, "F": 3.21},
            "medsam3": {"D": 67.15, "A": 13.98, "F": 15.24},
        },
        "MedSAM2": {
            "medsam2": {"D": 90.83, "A": 2.29, "F": 77.34},
            "ma_sam2": {"D": 90.85, "A": 2.28, "F": 77.37},
            "sam2long": {"D": 91.01, "A": 2.23, "F": 78.91},
            "echopilot": {"D": 95.31, "A": 0.83, "F": 77.35},
            "medsam3": {"D": 67.15, "A": 13.98, "F": 15.24},
        },
        "per_dataset_echopilot_sam2": {
            "camus": {"D": 34.09, "A": 28.86, "F": 3.21},
            "breast_lesion": {"D": 63.38, "A": 22.98, "F": 21.35},
            "placenta": {"D": 39.74, "A": 84.96, "F": 5.92},
        },
        "per_dataset_echopilot_medsam2": {
            "camus": {"D": 95.31, "A": 0.83, "F": 77.35},
            "breast_lesion": {"D": 68.44, "A": 28.20, "F": 23.91},
            "placenta": {"D": 38.87, "A": 63.40, "F": 3.28},
        },
    }


def table_stage1_ablation() -> dict[str, dict[str, float]]:
    """Table 2 — Stage I variants on CAMUS (MedSAM2 weights), Dice / ASD."""
    return {
        "medsam2_base": {"D": 90.83, "A": 2.29},
        "vfm_only": {"D": 78.32, "A": 2.65},
        "unimedclip_vfm": {"D": 84.69, "A": 2.08},
        "biomedclip_vfm": {"D": 95.31, "A": 0.83},
    }


def table_memory_gate_ablation() -> dict[str, float]:
    """Fig. 4 — Breast Lesion / MedSAM2: ASD with and without gate."""
    return {
        "without_gate_asd": 55.87,
        "with_gate_asd": 28.20,
        "tau_default": 0.5,
        "tau_stable_low": 0.1,
        "tau_stable_high": 0.5,
        "tau_aggressive": 0.9,
    }


def initialize_stage_i(
    height: int,
    width: int,
    user_point: tuple[int, int],
    category: str,
    cfg: EchoPilotConfig | None = None,
    *,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Stage I smoke: scale selection + prompt synthesis."""
    cfg = cfg or EchoPilotConfig()
    dev = torch.device(device)
    boxes = crop_boxes(height, width, user_point, cfg.scale_factors)

    # Synthetic VLM embeddings: middle scales match category best
    k = len(cfg.scale_factors)
    gen_i = torch.Generator(device=dev)
    gen_i.manual_seed(seed)
    gen_t = torch.Generator(device=dev)
    gen_t.manual_seed(seed + 1)
    img_embs = torch.randn(k, 64, device=dev, generator=gen_i)
    text_emb = torch.randn(64, device=dev, generator=gen_t)
    if "placenta" in category.lower():
        img_embs[2] = text_emb + 0.5 * torch.randn_like(text_emb)
        img_embs[1] = text_emb + 0.3 * torch.randn_like(text_emb)
    s_sem = semantic_similarity(img_embs, text_emb)

    feat = random_vfm_features(height, width, 32, device=dev, seed=seed + 2)
    sim = vfm_cosine_map(feat, (user_point[0], user_point[1]))
    s_spa_list = []
    for box in boxes:
        attr = stub_attribution_from_similarity(sim, box)
        s_spa_list.append(spatial_seed_score(attr))
    s_spa = torch.stack(s_spa_list)
    k_star, joint = select_scale_seed(s_sem, s_spa)
    box_star = boxes[k_star]
    prompts = synthesize_prompts(
        sim,
        user_point,
        box_star,
        max_aux=cfg.max_aux_prompts,
        nms_radius=cfg.nms_radius_px,
    )
    return {
        "category": category,
        "k_star": k_star,
        "scale_factor": cfg.scale_factors[k_star],
        "box": box_star,
        "prompts": prompts,
        "s_sem": [float(x) for x in s_sem],
        "s_spa": [float(x) for x in s_spa],
        "joint": [float(x) for x in joint],
    }


def propagate_stage_ii(
    anchor_mask: torch.Tensor,
    memory_features: list[torch.Tensor],
    cfg: EchoPilotConfig | None = None,
) -> dict[str, Any]:
    """Stage II smoke: reliability-gated writes over frame descriptors."""
    cfg = cfg or EchoPilotConfig()
    f0 = masked_descriptor(memory_features[0], anchor_mask)
    gates = gate_memory_sequence(
        [masked_descriptor(f, anchor_mask) for f in memory_features],
        f0,
        cfg.memory_gate_tau,
    )
    consistencies = [feature_consistency(masked_descriptor(f, anchor_mask), f0) for f in memory_features]
    return {
        "anchor_consistency": consistencies[0],
        "gates": gates,
        "write_rate": sum(gates) / len(gates),
        "tau": cfg.memory_gate_tau,
    }


def training_step_demo(
    cfg: EchoPilotConfig | None = None,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    cfg = cfg or EchoPilotConfig()
    h, w = 128, 128
    p0 = (64, 64)
    stage_i = initialize_stage_i(h, w, p0, cfg.category_default, cfg, device=device)

    gen = torch.Generator(device=torch.device(device))
    gen.manual_seed(7)
    mem = [torch.randn(h, w, 16, generator=gen, device=device) for _ in range(6)]
    mask = torch.zeros(h, w, device=device)
    mask[50:80, 50:80] = 1.0
    stage_ii = propagate_stage_ii(mask, mem, cfg)

    return {
        "num_prompts": float(len(stage_i["prompts"])),
        "k_star": float(stage_i["k_star"]),
        "write_rate": stage_ii["write_rate"],
        "first_gate": float(stage_ii["gates"][0]),
    }


def evaluation_demo(*, device: str = "cpu") -> dict[str, Any]:
    cfg = EchoPilotConfig()
    tab = table_main_results()
    abl = table_stage1_ablation()
    gate = table_memory_gate_ablation()
    step = training_step_demo(cfg, device=device)

    sam2_placenta_d = 16.74  # Table 1 Placenta / SAM2 baseline
    echo_placenta = tab["per_dataset_echopilot_sam2"]["placenta"]
    dice_gain = echo_placenta["D"] - sam2_placenta_d

    return {
        **step,
        "camus_dice_medsam2": tab["MedSAM2"]["echopilot"]["D"],
        "camus_asd_medsam2": tab["MedSAM2"]["echopilot"]["A"],
        "placenta_dice_gain_sam2": dice_gain,
        "biomedclip_vs_base_dice": abl["biomedclip_vfm"]["D"] - abl["medsam2_base"]["D"],
        "gate_asd_reduction": gate["without_gate_asd"] - gate["with_gate_asd"],
        "beats_medsam3_placenta": echo_placenta["D"] > 20.69,
    }
