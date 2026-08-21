"""SegCompass framework card, paper tables, and smoke demos (arXiv:2605.22658)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.segcompass.activation import avg_active_features, instance_coverage_topk
from ltx_trainer.segcompass.alignment import paradigm_properties
from ltx_trainer.segcompass.codebook import aggregate_concept_slots, decode_sparse_pairs
from ltx_trainer.segcompass.config import MLLMBackbone, SAE_LAYERS, SegCompassConfig
from ltx_trainer.segcompass.decoder import decode_masks
from ltx_trainer.segcompass.encoder import encode_slot_concepts
from ltx_trainer.segcompass.grpo import clipped_policy_ratio, group_advantages
from ltx_trainer.segcompass.heatmap import multi_slot_heatmaps
from ltx_trainer.segcompass.layout import LIMITATIONS
from ltx_trainer.segcompass.matching import hungarian_mean_overlap
from ltx_trainer.segcompass.metrics import cumulative_iou, generalized_iou
from ltx_trainer.segcompass.mock import toy_reasoning_instruction, toy_sparse_activations
from ltx_trainer.segcompass.objectives import confidence_loss, segmentation_loss, total_objective
from ltx_trainer.segcompass.policy import mcot_rollout_spec, policy_logprob_ratio
from ltx_trainer.segcompass.rewards import (
    combined_reward,
    format_score,
    multi_object_mask_reward,
    soft_iou,
)
from ltx_trainer.segcompass.sae import sae_reconstruction_loss, sparsity_fraction, support_indices
from ltx_trainer.segcompass.slot_mapper import (
    fuse_slot_query,
    multi_head_attention_scores,
    slot_confidence,
)


def framework_card(cfg: SegCompassConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SegCompassConfig()
    return {
        "name": "SegCompass",
        "paper": cfg.paper_arxiv,
        "code": cfg.code_url,
        "idea": (
            "SAE-driven interpretable alignment between CoT reasoning and segmentation: "
            "sparse concept space → query codebook → multi-slot heatmap → SAM-style decoder, "
            "trained with GRPO on language and supervised mask losses."
        ),
        "alignment_paradigm": "interpretable_sparse_concepts",
        "baselines_contrast": ["latent_query_alignment", "textual_localization_readout"],
        "mllm_backbones": [b.value for b in MLLMBackbone],
        "sae": {"d_pi": cfg.policy_hidden_d, "d_sae": cfg.sae_dim, "d_concept": cfg.concept_dim},
        "training": {
            "grpo_group_size": cfg.grpo_group_size,
            "lambda_seg": cfg.lambda_seg,
            "lambda_conf": cfg.lambda_conf,
            "reward_weights": {"format": cfg.reward_format, "segmentation": cfg.reward_segmentation},
        },
        "vision_backbone": cfg.vision_backbone,
        "sae_extract_layers": [
            {"backbone": L.backbone, "layer": L.layer_index, "total_layers": L.num_layers}
            for L in SAE_LAYERS
        ],
        "alignment_paradigms": paradigm_properties(),
        "benchmarks": ["RefCOCO", "RefCOCO+", "RefCOCOg", "gRefCOCO", "ReasonSeg"],
    }


def table_refcoco_full() -> list[dict[str, str | float]]:
    """Table 1 — full SegCompass-13B1 split coverage."""
    return [
        {
            "method": "SegCompass-13B1",
            "refcoco_val": 86.3,
            "refcoco_testA": 87.3,
            "refcoco_testB": 84.1,
            "refcoco+_val": 80.5,
            "refcoco+_testA": 84.6,
            "refcoco+_testB": 76.9,
            "refcocog_val": 84.0,
            "refcocog_test": 81.3,
        },
    ]


def table_refcoco_series() -> list[dict[str, str | float]]:
    """Table 1 — RefCOCO / RefCOCO+ / RefCOCOg (selected rows)."""
    return [
        {"method": "LISA-7B", "refcoco_val": 74.9, "refcoco_testA": 79.1, "refcoco+_val": 65.1, "refcocog_val": 67.9},
        {"method": "HiMTok-8B", "refcoco_val": 85.9, "refcoco_testA": 86.3, "refcoco+_val": 80.5, "refcocog_val": 80.1},
        {"method": "X-SAM-3.8B", "refcoco_val": 85.1, "refcoco_testA": 87.1, "refcoco+_val": 78.0, "refcocog_val": 83.8},
        {"method": "SegCompass-7B1", "refcoco_val": 80.0, "refcoco_testA": 82.9, "refcoco+_val": 77.2, "refcocog_val": 79.4},
        {"method": "SegCompass-7B2", "refcoco_val": 85.3, "refcoco_testA": 87.5, "refcoco+_val": 78.3, "refcocog_val": 82.8},
        {"method": "SegCompass-13B1", "refcoco_val": 86.3, "refcoco_testA": 87.3, "refcoco+_val": 80.5, "refcocog_val": 84.0},
    ]


def table_grefcoco() -> list[dict[str, str | float]]:
    """Table 2 — gRefCOCO multi-object."""
    return [
        {"method": "LISA-7B", "val_giou": 61.6, "val_ciou": 61.7, "testA_ciou": 68.5},
        {"method": "HiMTok-8B", "val_giou": 72.1, "val_ciou": 70.4, "testA_ciou": 74.9},
        {"method": "SegCompass-7B2", "val_giou": 76.1, "val_ciou": 72.2, "testA_ciou": 76.5},
        {"method": "SegCompass-13B1", "val_giou": 76.8, "val_ciou": 72.2, "testA_ciou": 77.3},
    ]


def table_reasonseg_zero_shot() -> list[dict[str, str | float]]:
    """Table 3 — ReasonSeg zero-shot."""
    return [
        {"method": "LISA-7B", "val_giou": 53.6, "val_ciou": 52.3, "test_ciou": 48.8},
        {"method": "Seg-Zero-7B", "val_giou": 62.6, "val_ciou": 62.0, "test_ciou": 52.0},
        {"method": "VisionReasoner-7B", "val_giou": 66.3, "val_ciou": 0.0, "test_ciou": 0.0},
        {"method": "SegCompass-7B2", "val_giou": 66.6, "val_ciou": 66.3, "test_ciou": 64.8},
        {"method": "SegCompass-13B1", "val_giou": 67.4, "val_ciou": 67.2, "test_ciou": 66.5},
    ]


def table_training_mode_ablation() -> list[dict[str, str | float | bool]]:
    """Table 4 — RL vs segmentation supervision."""
    return [
        {"rl": True, "seg_sup": False, "refcocog": 65.9, "grefcoco": 63.0, "reasonseg": 40.1},
        {"rl": False, "seg_sup": True, "refcocog": 77.9, "grefcoco": 74.0, "reasonseg": 59.3},
        {"rl": True, "seg_sup": True, "refcocog": 81.3, "grefcoco": 77.3, "reasonseg": 66.5},
    ]


def table_vision_backbone_ablation() -> list[dict[str, str | float]]:
    """Table 5 — ViT-B / L / H."""
    return [
        {"backbone": "ViT-B", "params_B": 0.09, "refcocog": 78.8, "grefcoco": 75.0, "reasonseg": 58.8},
        {"backbone": "ViT-L", "params_B": 0.31, "refcocog": 80.1, "grefcoco": 76.2, "reasonseg": 63.9},
        {"backbone": "ViT-H", "params_B": 0.64, "refcocog": 81.3, "grefcoco": 77.3, "reasonseg": 66.5},
    ]


def table_reward_ablation() -> list[dict[str, float]]:
    """Table 6 — format vs segmentation reward weights."""
    return [
        {"format": 0.0, "segmentation": 1.0, "refcocog": 79.1, "grefcoco": 75.0, "reasonseg": 63.9},
        {"format": 0.5, "segmentation": 0.5, "refcocog": 81.0, "grefcoco": 76.8, "reasonseg": 66.2},
        {"format": 0.3, "segmentation": 0.7, "refcocog": 81.3, "grefcoco": 77.3, "reasonseg": 66.5},
    ]


def table_grpo_group_size() -> dict[int, dict[str, float]]:
    """Fig. 6 — cIoU vs GRPO group size G (SegCompass-13B)."""
    return {
        2: {"refcocog": 76.9, "grefcoco": 72.6, "reasonseg": 61.9},
        4: {"refcocog": 79.2, "grefcoco": 75.0, "reasonseg": 64.0},
        6: {"refcocog": 80.4, "grefcoco": 76.8, "reasonseg": 65.1},
        8: {"refcocog": 81.3, "grefcoco": 77.3, "reasonseg": 66.5},
    }


def interpretability_correlations() -> dict[str, float]:
    """Fig. 8 — OLS correlations on gRefCOCO (Qwen2.5-VL-7B)."""
    return {
        "sae_mse_vs_dice_loss_R": 0.68,
        "heatmap_ciou_vs_mask_ciou_R": 0.69,
        "heatmap_ciou_vs_mask_ciou_R_extended_n": 0.79,
    }


def training_stack() -> dict[str, Any]:
    """Appendix — distributed RL + segmentation training stack."""
    return {
        "rl_framework": "VERL",
        "policy_parallelism": "FSDP",
        "rollout_engine": "vLLM",
        "vision_decoder": "SAM-ViT-H",
        "sae_trained_offline": True,
        "grpo_group_size_default": 8,
    }


def training_curriculum(cfg: SegCompassConfig | None = None) -> dict[str, Any]:
    """Two-stage schedule: offline SAE pretrain → RL+seg joint finetune."""
    cfg = cfg or SegCompassConfig()
    return {
        "stage1_sae": {
            "dataset": "OBELICS",
            "samples": cfg.obelics_sae_samples,
            "frozen_at_inference": True,
        },
        "stage2_joint": {
            "steps": cfg.training_steps,
            "grpo_group_size": cfg.grpo_group_size,
            "lr_mllm": cfg.mllm_lr,
            "lr_multipliers": {
                "codebook": cfg.lr_multiplier_codebook,
                "slot_mapper": cfg.lr_multiplier_slot_mapper,
                "mask_decoder": cfg.lr_multiplier_mask_decoder,
            },
        },
    }


def interpretability_analysis(cfg: SegCompassConfig | None = None) -> dict[str, Any]:
    """Fig. 4–5 activation patterns + Fig. 8 correlations."""
    cfg = cfg or SegCompassConfig()
    return {
        "correlations": interpretability_correlations(),
        "activation_by_token_type": {
            "instance_tau01": avg_active_features("instance", 0.1, backbone="qwen2.5-vl-7b"),
            "background_tau01": avg_active_features("background", 0.1, backbone="qwen2.5-vl-7b"),
            "reasoning_tau01": avg_active_features("cot", 0.1, backbone="qwen2.5-vl-7b"),
        },
        "instance_coverage_top10pct": instance_coverage_topk(0.1),
        "sae_dim": cfg.sae_dim,
    }


def pipeline_demo(cfg: SegCompassConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SegCompassConfig()
    h = toy_sparse_activations(n_active=12, dim=min(cfg.sae_dim, 1024))
    support = support_indices(h)
    h_l1 = sum(abs(h[j]) for j in support)
    sae_loss = sae_reconstruction_loss(1.0, 0.92, h_l1, alpha=cfg.sae_alpha)
    decoded = decode_sparse_pairs(support, [h[j] for j in support], concept_dim=cfg.concept_dim)
    slots = aggregate_concept_slots(decoded, num_slots=cfg.max_slots)
    encoded = encode_slot_concepts(slots)
    slot_repr = encoded[0] if encoded else []
    q = fuse_slot_query(0.8, sum(slot_repr[:4]) if slot_repr else 0.0)
    mha = multi_head_attention_scores([q], [0.7, 0.5, 0.3])
    spatial_heatmaps = multi_slot_heatmaps(mha)
    conf = slot_confidence(max(mha[0]) if mha and mha[0] else 0.0)
    heatmap_peaks = [max(hm) if hm else 0.0 for hm in spatial_heatmaps]
    keys = [0.75, 0.68]
    confidences = [conf, conf * 0.9]
    masks = decode_masks(keys, heatmap_peaks or [0.82, 0.71], confidences)
    multi_r = multi_object_mask_reward(masks, [0.80, 0.70])
    instruction = toy_reasoning_instruction()
    rollout = mcot_rollout_spec(instruction, num_slots=cfg.max_slots)
    log_ratio = policy_logprob_ratio(0.55, 0.50)
    seg = segmentation_loss(0.85, 0.78, lambda_dice=cfg.lambda_dice)
    conf_l = confidence_loss(conf, matched=True)
    total = total_objective(0.1, seg, conf_l, lambda_s=cfg.lambda_seg, lambda_c=cfg.lambda_conf)
    sample_response = (
        "<think>Identify the white ceramic bowl and sandwich halves.</think>"
        "<REF>"
    )
    fmt = format_score(sample_response)
    mask_r = soft_iou(0.82, 0.79)
    reward = combined_reward(fmt, mask_r, w_format=cfg.reward_format, w_seg=cfg.reward_segmentation)
    advantages = group_advantages([reward, 0.4, 0.6, 0.55, 0.5, 0.45, 0.7, 0.65])
    grpo_term = clipped_policy_ratio(0.9, 0.8, advantages[0], epsilon=cfg.grpo_clip_ratio)
    ciou = cumulative_iou(masks[0], 0.79) if masks else 0.0
    giou = generalized_iou(masks, [0.80, 0.70])
    return {
        "support_size": len(support),
        "sparsity_fraction": sparsity_fraction(len(support), len(h)),
        "sae_loss_scalar": sae_loss,
        "num_concept_slots": len(slots),
        "encoded_slot_dim": len(slot_repr),
        "num_spatial_heatmaps": len(spatial_heatmaps),
        "slot_confidence": conf,
        "decoded_masks": masks,
        "cumulative_iou": ciou,
        "generalized_iou": giou,
        "multi_object_mask_reward": multi_r,
        "mcot_concentration_tokens": rollout.num_concentration_tokens,
        "policy_logprob_ratio": log_ratio,
        "hungarian_overlap": hungarian_mean_overlap(masks, [0.79, 0.72]),
        "format_reward": fmt,
        "combined_rollout_reward": reward,
        "grpo_advantage_first": advantages[0] if advantages else 0.0,
        "grpo_clipped_term": grpo_term,
        "total_loss_scalar": total,
        "instance_coverage_top10pct": instance_coverage_topk(0.1),
    }


def evaluation_demo(cfg: SegCompassConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SegCompassConfig()
    best = next(r for r in table_refcoco_series() if r["method"] == "SegCompass-13B1")
    lisa = next(r for r in table_refcoco_series() if r["method"] == "LISA-7B")
    delta_refcocog = float(best["refcocog_val"]) - float(lisa["refcocog_val"])
    return {
        "framework": framework_card(cfg),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "refcocog_ciou_gain_vs_lisa_7b": round(delta_refcocog, 1),
        "interpretability": interpretability_analysis(cfg),
        "training_stack": training_stack(),
        "training_curriculum": training_curriculum(cfg),
        "paper_tables": {
            "refcoco": table_refcoco_series(),
            "refcoco_full": table_refcoco_full(),
            "grefcoco": table_grefcoco(),
            "reasonseg": table_reasonseg_zero_shot(),
            "training_mode": table_training_mode_ablation(),
            "vision_backbone": table_vision_backbone_ablation(),
            "reward_ablation": table_reward_ablation(),
            "grpo_group_size": table_grpo_group_size(),
        },
        "activation_patterns": interpretability_analysis(cfg)["activation_by_token_type"],
    }
