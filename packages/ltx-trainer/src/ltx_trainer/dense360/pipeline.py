"""Training / ablation demos."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.dense360.benchmarks import TABLE4_ABLATION
from ltx_trainer.dense360.config import Dense360Config
from ltx_trainer.dense360.dense360_vlm import Dense360VLMStub
from ltx_trainer.dense360.metrics import caption_phrase_recall, mask_iou
from ltx_trainer.dense360.slicing import erp_window_slices
from ltx_trainer.dense360.synthetic import synthetic_entity_mask, synthetic_erp


def evaluation_demo_run(cfg: Dense360Config | None = None) -> dict[str, Any]:
    cfg = cfg or Dense360Config(height=128, width=256)
    model = Dense360VLMStub(cfg)
    erp = synthetic_erp(cfg)
    with torch.no_grad():
        out = model(erp, referring_text="person in red jacket at center")
    return {
        "erp_shape": list(erp.shape),
        "seg_shape": list(out["seg_logits"].shape),
        "erp_rope": model.erp_rope_demo(),
    }


def ablation_erp_rope(cfg: Dense360Config | None = None) -> dict[str, float]:
    cfg = cfg or Dense360Config(height=64, width=128)
    erp = synthetic_erp(cfg)
    with_rope = Dense360VLMStub(cfg)
    no_rope = Dense360VLMStub(replace(cfg, use_erp_rope=False))
    with torch.no_grad():
        a = with_rope(erp)["seg_logits"].mean()
        b = no_rope(erp)["seg_logits"].mean()
    return {
        "with_rope_mean": float(a),
        "no_rope_mean": float(b),
        "table4_full_caption": TABLE4_ABLATION["full_with_rope"]["caption_omni"],
        "table4_no_rope_caption": TABLE4_ABLATION["full_no_rope"]["caption_omni"],
    }


def slicing_demo(cfg: Dense360Config | None = None) -> dict[str, Any]:
    cfg = cfg or Dense360Config(height=64, width=256)
    erp = synthetic_erp(cfg)
    slices = erp_window_slices(erp)
    return {"num_slices": len(slices), "slice_shapes": [list(s.shape) for s in slices]}


def train_step(cfg: Dense360Config | None = None) -> dict[str, float]:
    cfg = cfg or Dense360Config(height=64, width=128)
    model = Dense360VLMStub(cfg)
    erp = synthetic_erp(cfg)
    gt = synthetic_entity_mask(cfg)
    out = model(erp)
    pred = torch.sigmoid(out["seg_logits"])
    iou = float(mask_iou(pred, gt).mean().detach())
    recall = caption_phrase_recall(
        "A person in a red jacket standing near the center of the room.",
        ["red jacket", "center", "person"],
    )
    loss = F.binary_cross_entropy_with_logits(out["seg_logits"], gt)
    return {
        "seg_bce": float(loss.detach()),
        "mask_iou": iou,
        "phrase_recall": recall,
    }
