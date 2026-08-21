"""HD LoRA attention theory stub (Duranthon et al., arXiv:2606.05899)."""

from ltx_trainer.lora_hd_attn.config import LoraHdAttnConfig
from ltx_trainer.lora_hd_attn.mock import evaluation_smoke
from ltx_trainer.lora_hd_attn.paper import knowledge_bundle, paper_card
from ltx_trainer.lora_hd_attn.pipeline import run_active_ft_demo, run_demo, run_reused_sequences_demo

__all__ = [
    "LoraHdAttnConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_active_ft_demo",
    "run_demo",
    "run_reused_sequences_demo",
]
