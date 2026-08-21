"""Framework card, Tables 1–5, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.ea_cot.binding import decompose_total_gain, s2t_gap
from ltx_trainer.ea_cot.config import EACoTConfig
from ltx_trainer.ea_cot.model import EntityBindingStub, binding_bce
from ltx_trainer.ea_cot.prompts import EA_COT_STAGES, ea_cot_prompt


def framework_card(cfg: EACoTConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EACoTConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "benchmark": cfg.benchmark,
        "categories": list(cfg.categories),
        "models": list(cfg.models),
        "intervention": "Entity-Aware Chain-of-Thought (EA-CoT)",
        "diagnosis": "entity binding failure under speech encoding (pooling blurs token boundaries)",
        "token_budget": {"baseline": cfg.baseline_tokens, "ea_cot": cfg.ea_cot_tokens},
        "ea_cot_stages": list(EA_COT_STAGES),
    }


def table1_bbh_accuracy() -> list[dict[str, Any]]:
    """Table 1 — BBH accuracy S2T / T2T (%)."""
    return [
        {
            "model": "Qwen2.5-Omni",
            "method": "BL",
            "overall": "59.9 / 67.0",
            "HYP": "73.2 / 72.0",
            "NAV": "58.0 / 52.8",
            "SPO": "55.6 / 56.4",
            "WOL": "52.8 / 86.8",
        },
        {
            "model": "Qwen2.5-Omni",
            "method": "CoT",
            "overall": "68.5 / 84.3",
            "HYP": "62.4 / 83.2",
            "NAV": "80.4 / 80.8",
            "SPO": "61.6 / 77.6",
            "WOL": "69.6 / 95.6",
        },
        {
            "model": "Phi-4-MM",
            "method": "BL",
            "overall": "53.6 / 66.7",
            "HYP": "56.4 / 61.6",
            "NAV": "59.2 / 58.0",
            "SPO": "48.0 / 55.6",
            "WOL": "50.8 / 91.6",
        },
        {
            "model": "Phi-4-MM",
            "method": "CoT",
            "overall": "62.7 / 77.6",
            "HYP": "54.8 / 77.2",
            "NAV": "66.4 / 82.0",
            "SPO": "54.4 / 64.0",
            "WOL": "75.2 / 87.2",
        },
    ]


def table2_wol_ablation() -> list[dict[str, Any]]:
    """Table 2 — Qwen S2T web of lies ablation."""
    return [
        {"component": "Baseline", "acc": 51.6, "delta_pp": None, "pct_of_full": None},
        {"component": "+ Format only", "acc": 55.6, "delta_pp": 4.0, "pct_of_full": 23},
        {"component": "+ Step-by-step", "acc": 59.2, "delta_pp": 7.6, "pct_of_full": 43},
        {"component": "+ Entity enum", "acc": 62.0, "delta_pp": 10.4, "pct_of_full": 59},
        {"component": "Full EA-CoT", "acc": 69.2, "delta_pp": 17.6, "pct_of_full": 100},
    ]


def table3_entity_corruption() -> list[dict[str, Any]]:
    """Table 3 — T2T name corruption on web of lies (Qwen)."""
    return [
        {"corruption_pct": 0, "bl": 89.6, "ea_cot": 95.6, "delta": 6.0},
        {"corruption_pct": 25, "bl": 89.2, "ea_cot": 94.4, "delta": 5.2},
        {"corruption_pct": 50, "bl": 84.8, "ea_cot": 93.2, "delta": 8.4},
        {"corruption_pct": 75, "bl": 88.4, "ea_cot": 94.0, "delta": 5.6},
        {"corruption_pct": 100, "bl": 86.0, "ea_cot": 92.0, "delta": 6.0},
    ]


def table5_bbh_vs_mmsu() -> list[dict[str, Any]]:
    """Table 5 — semantic BBH vs acoustic MMSU modality gaps."""
    return [
        {"model": "Qwen", "bbh_s2t": 59.9, "bbh_t2t": 67.0, "bbh_gap": -7.1, "mmsu_s2t": 80.1, "mmsu_t2t": 49.0, "mmsu_gap": 31.0},
        {"model": "Phi-4", "bbh_s2t": 53.6, "bbh_t2t": 66.7, "bbh_gap": -13.1, "mmsu_s2t": 71.9, "mmsu_t2t": 47.5, "mmsu_gap": 24.5},
    ]


def wol_recovery_gains() -> dict[str, float]:
    """Figure 2 / §4.2 — web of lies S2T CoT gains (pp)."""
    return {"Qwen2.5-Omni": 16.8, "Phi-4-MM": 24.4}


def token_budget_qwen_speech() -> dict[str, float]:
    """§4.3 — Qwen speech: budget ~0, instruction drives gain."""
    return decompose_total_gain(bl256=52.8, bl1024=52.8, cot1024=69.6)


def forward_smoke(cfg: EACoTConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EACoTConfig()
    model = EntityBindingStub(cfg)
    b = 2
    speech = torch.randn(b, cfg.demo_claim_len)
    entities = torch.randint(0, 8, (b, 4))
    claims = torch.randn(b, cfg.demo_claim_len)
    logits = model(speech, entities, claims)
    loss = float(binding_bce(logits, torch.tensor([1.0, 0.0])).detach())
    wol = [r for r in table1_bbh_accuracy() if r["method"] == "CoT" and "Phi" in r["model"]][0]
    gap_bl = s2t_gap(50.8, 91.6)
    return {
        "logits_shape": list(logits.shape),
        "binding_loss": loss,
        "ea_cot_prompt_lines": len(ea_cot_prompt().splitlines()),
        "wol_phi_bl_gap_pp": gap_bl,
        "wol_phi_cot_s2t": float(wol["WOL"].split("/")[0].strip()),
        "token_decomp": token_budget_qwen_speech(),
    }


def evaluation_demo(cfg: EACoTConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EACoTConfig()
    return {
        "framework": framework_card(cfg),
        "table1": table1_bbh_accuracy(),
        "table2_ablation": table2_wol_ablation(),
        "table3_corruption": table3_entity_corruption(),
        "table5_bbh_mmsu": table5_bbh_vs_mmsu(),
        "wol_recovery_pp": wol_recovery_gains(),
        "token_budget_speech": token_budget_qwen_speech(),
        "forward": forward_smoke(cfg),
    }
