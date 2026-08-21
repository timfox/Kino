"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dense360.benchmarks import TABLE3_BENCH, benchmarks_bundle
from ltx_trainer.dense360.config import PAPER_ARXIV, Dense360Config
from ltx_trainer.dense360.datasets import datasets_card
from ltx_trainer.dense360.paper import framework_card
from ltx_trainer.dense360.pipeline import (
    ablation_erp_rope,
    evaluation_demo_run,
    slicing_demo,
    train_step,
)
from ltx_trainer.dense360.dense360_vlm import Dense360VLMStub
from ltx_trainer.dense360.synthetic import synthetic_erp


def evaluation_smoke() -> dict[str, Any]:
    cfg = Dense360Config(height=128, width=256)
    model = Dense360VLMStub(cfg)
    erp = synthetic_erp(cfg)
    out = model(erp)
    ours = TABLE3_BENCH["Dense360VLM-3B"]
    sa2va = TABLE3_BENCH["SA2VA-4B_dense360"]
    qwen = TABLE3_BENCH["Qwen2.5VL-72B"]

    return {
        "package": "dense360",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "caption_omni": ours["caption"]["omni"],
        "grounding_omni": ours["grounding"]["omni"],
        "beats_sa2va_caption": ours["caption"]["omni"] > sa2va["caption"]["omni"],
        "beats_sa2va_grounding": ours["grounding"]["omni"] > sa2va["grounding"]["omni"],
        "beats_qwen_grounding": ours["grounding"]["omni"] > qwen["grounding"]["omni"],
        "back_grounding": ours["grounding"]["back"],
        "seg_shape": list(out["seg_logits"].shape),
        "train": train_step(cfg),
        "demo": evaluation_demo_run(cfg),
        "ablation": ablation_erp_rope(cfg),
        "slicing": slicing_demo(cfg),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "table3_methods": list(TABLE3_BENCH.keys()),
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
