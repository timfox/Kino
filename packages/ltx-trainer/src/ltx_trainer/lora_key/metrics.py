"""Paper tables — § V."""

from __future__ import annotations


def table_i_fidelity_watermark() -> list[dict[str, str | float | None]]:
    """Table I — content + style rows."""
    content = [
        {"task": "Content", "method": "None", "fid": 17.53, "clip": 30.15, "dreamsim": None, "bit_acc": None, "bit_acc_adv": None, "tpr": None, "tpr_adv": None},
        {"task": "Content", "method": "AuthenLoRA", "fid": 19.85, "clip": 25.10, "dreamsim": 0.24, "bit_acc": 94.27, "bit_acc_adv": 92.15, "tpr": 0.91, "tpr_adv": 0.84},
        {"task": "Content", "method": "Ours", "fid": 18.72, "clip": 29.85, "dreamsim": 0.20, "bit_acc": 98.23, "bit_acc_adv": 97.85, "tpr": 1.00, "tpr_adv": 1.00},
    ]
    style = [
        {"task": "Style", "method": "AuthenLoRA", "fid": 28.90, "clip": 22.80, "dreamsim": 0.25, "bit_acc": 94.93, "bit_acc_adv": 91.04, "tpr": 0.93, "tpr_adv": 0.81},
        {"task": "Style", "method": "Ours", "fid": 20.55, "clip": 26.80, "dreamsim": 0.19, "bit_acc": 99.96, "bit_acc_adv": 99.71, "tpr": 1.00, "tpr_adv": 1.00},
    ]
    return content + style


def table_ii_robustness() -> list[dict[str, str | float]]:
    """Table II — perturbation bit accuracy (content ours + style ours excerpt)."""
    return [
        {"task": "Content", "method": "Ours", "resize": 98.20, "g_blur": 98.21, "g_noise": 95.77, "jpeg": 97.89, "avg": 97.85},
        {"task": "Content", "method": "AuthenLoRA", "resize": 88.45, "g_blur": 89.12, "g_noise": 86.70, "jpeg": 91.35, "avg": 92.15},
        {"task": "Style", "method": "Ours", "resize": 99.95, "g_blur": 99.95, "g_noise": 98.25, "jpeg": 99.85, "avg": 99.71},
    ]


def table_iii_scalability() -> list[dict[str, str | float | int]]:
    """Table III — LoRA-Key vs AuthenLoRA scaling."""
    return [
        {"method": "LoRA-Key", "loras": 1, "tpr": 1.00, "tpr_adv": 1.00, "train_h": 4.82, "attach_h": 0.03, "total_h": 4.85, "avg_per_lora_h": 4.85},
        {"method": "LoRA-Key", "loras": 10, "tpr": 0.99, "tpr_adv": 0.99, "train_h": 4.82, "attach_h": 0.28, "total_h": 5.10, "avg_per_lora_h": 0.51},
        {"method": "AuthenLoRA", "loras": 1, "tpr": 0.93, "tpr_adv": 0.81, "train_h": 15.37, "attach_h": 0.0, "total_h": 15.37, "avg_per_lora_h": 15.37},
        {"method": "AuthenLoRA", "loras": 10, "tpr": 0.91, "tpr_adv": 0.81, "train_h": 153.70, "attach_h": 0.0, "total_h": 153.70, "avg_per_lora_h": 15.37},
    ]


def table_iv_cross_architecture() -> list[dict[str, str | float]]:
    return [
        {"model": "SD 1.4", "fid": 18.72, "bit_acc": 98.23, "bit_acc_adv": 97.86, "tpr": 1.00, "tpr_adv": 1.00},
        {"model": "SDXL", "fid": 17.72, "bit_acc": 96.27, "bit_acc_adv": 91.88, "tpr": 1.00, "tpr_adv": 0.87},
        {"model": "PixArt-α", "fid": 19.49, "bit_acc": 99.24, "bit_acc_adv": 99.09, "tpr": 1.00, "tpr_adv": 1.00},
    ]


def table_v_community_lora() -> list[dict[str, str | float]]:
    return [
        {"domain": "Anime Character", "fid": 27.41, "bit_acc": 94.63, "bit_acc_adv": 92.11},
        {"domain": "Realistic Portrait", "fid": 25.93, "bit_acc": 97.12, "bit_acc_adv": 95.48},
        {"domain": "Architecture", "fid": 30.77, "bit_acc": 97.88, "bit_acc_adv": 96.63},
        {"domain": "Average", "fid": 28.38, "bit_acc": 96.14, "bit_acc_adv": 94.52},
    ]


def table_vii_rank_ablation() -> list[dict[str, int | float]]:
    return [
        {"style_rank": 64, "fid": 18.72, "bit_acc": 98.23, "tpr": 1.00},
        {"style_rank": 320, "fid": 23.47, "bit_acc": 95.80, "tpr": 0.99},
        {"style_rank": 32, "fid": 29.50, "bit_acc": 95.18, "tpr": 0.99},
    ]


def gop_cosine_ablation() -> dict[str, float]:
    """Fig. 4 — with/without GOP mean LoRA-key cosine similarity."""
    return {"without_gop_mean_cosine": 0.92, "with_gop_mean_cosine": 0.04}
