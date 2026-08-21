"""Framework card, Tables 1–5, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.iwslt26_if.config import IWSLT26IFConfig
from ltx_trainer.iwslt26_if.model import SpeechIFStub
from ltx_trainer.iwslt26_if.rerank import lik_mbr_hybrid, mbr_select
from ltx_trainer.iwslt26_if.sampling import table1_distribution


def framework_card(cfg: IWSLT26IFConfig | None = None) -> dict[str, Any]:
    cfg = cfg or IWSLT26IFConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "dataset_hub": cfg.dataset_hub,
        "tracks": list(cfg.tracks),
        "languages": list(cfg.languages),
        "tasks": list(cfg.tasks),
        "training": {
            "samples": cfg.total_samples,
            "interleave_T": cfg.interleave_temperature,
            "lora_rank": cfg.lora_rank,
            "max_tokens": cfg.max_tokens,
        },
        "submissions": {
            "primary": cfg.primary_model,
            "contrastive": f"{cfg.contrastive_asr} + {cfg.contrastive_llm}",
            "rerank_en_zh": cfg.rerank_en_zh,
        },
    }


def table2_mcif_long_primary() -> dict[str, Any]:
    """Row (7) N2+IT+Avg — primary submission (fixed prompt, macro-avg)."""
    return {
        "model": "N2+IT+Avg (Primary)",
        "SQA_fix": 40.42,
        "SSUM_fix": 26.10,
        "ASR_fix": 37.65,
        "ST_fix": 73.94,
        "SQA_mix": 40.31,
        "SSUM_mix": 29.41,
        "ASR_mix": 36.88,
        "ST_mix": 76.01,
    }


def table2_cascaded_contrastive() -> dict[str, Any]:
    """Row (10) Cascaded+FT."""
    return {
        "model": "Cascaded+FT",
        "SQA_fix": 33.36,
        "SSUM_fix": 28.55,
        "ASR_fix": 5.90,
        "ST_fix": 83.72,
    }


def table3_reranking_delta() -> list[dict[str, Any]]:
    """Table 3 — Δ vs greedy (macro-avg)."""
    return [
        {"method": "Greedy", "ASR": 40.77, "SQA": 37.24, "SSUM": 29.38, "ST": 75.28, "impr": None},
        {"method": "Likelihood", "ASR": -24.93, "SQA": -11.06, "SSUM": -8.60, "ST": 0.93, "impr": 1.55},
        {"method": "MBR", "ASR": 3.09, "SQA": 0.22, "SSUM": -0.79, "ST": 0.27, "impr": -0.85},
        {"method": "Lik. + MBR", "ASR": -19.28, "SQA": -3.33, "SSUM": -2.19, "ST": 1.09, "impr": 3.71},
    ]


def table4_en_zh_rerank() -> dict[str, dict[str, float]]:
    """Table 4 — primary vs reranked on EN/ZH fixed prompts."""
    return {
        "EN": {
            "SQA_base": 44.91,
            "SQA_rerank": 44.30,
            "SSUM_base": 23.44,
            "SSUM_rerank": 23.10,
            "ASR_base": 37.65,
            "ASR_rerank": 21.39,
        },
        "ZH": {
            "SQA_base": 39.31,
            "SQA_rerank": 40.52,
            "SSUM_base": 39.57,
            "SSUM_rerank": 40.11,
            "ST_base": 79.06,
            "ST_rerank": 80.61,
        },
    }


def table5_official_iwslt() -> list[dict[str, Any]]:
    """Table 5 — official IWSLT 2026 averages."""
    return [
        {
            "track": "Short",
            "submission": "Primary",
            "COMET": 0.844,
            "SQA_BERT": 0.484,
            "QE_Acc": 0.000,
            "ASR_WER": 0.074,
        },
        {
            "track": "Short",
            "submission": "Contrastive",
            "COMET": 0.838,
            "SQA_BERT": 0.448,
            "QE_Acc": 0.722,
            "ASR_WER": 0.170,
        },
        {
            "track": "Long",
            "submission": "Primary",
            "COMET": 0.751,
            "SQA_BERT": 0.427,
            "QE_Acc": 0.000,
            "ASR_WER": 0.269,
            "SSUM_BERT": 0.275,
            "ACHAP_F1": 0.474,
        },
        {
            "track": "Long",
            "submission": "Contrastive",
            "COMET": 0.843,
            "SQA_BERT": 0.344,
            "QE_Acc": 0.722,
            "ASR_WER": 0.064,
            "SSUM_BERT": 0.268,
            "ACHAP_F1": 0.421,
        },
    ]


def forward_smoke(cfg: IWSLT26IFConfig | None = None) -> dict[str, Any]:
    cfg = cfg or IWSLT26IFConfig()
    model = SpeechIFStub(cfg)
    audio = torch.randn(2, cfg.demo_audio_len)
    text = torch.randint(0, 256, (2, cfg.demo_text_len))
    logits = model(audio, text)
    cands = ["greedy transcript", "shas segmented", "sample variant"]
    lp = torch.tensor([-1.2, -0.8, -1.5])
    pick = lik_mbr_hybrid(cands, lp)
    return {
        "logits_shape": list(logits.shape),
        "table1_rows": len(table1_distribution()),
        "mbr_index": mbr_select(cands),
        "rerank_pick": pick,
        "interleave_T": cfg.interleave_temperature,
    }


def evaluation_demo(cfg: IWSLT26IFConfig | None = None) -> dict[str, Any]:
    cfg = cfg or IWSLT26IFConfig()
    return {
        "framework": framework_card(cfg),
        "table1": table1_distribution(cfg.interleave_temperature),
        "table2_primary": table2_mcif_long_primary(),
        "table2_cascaded": table2_cascaded_contrastive(),
        "table3_rerank": table3_reranking_delta(),
        "table4_en_zh": table4_en_zh_rerank(),
        "table5_official": table5_official_iwslt(),
        "forward": forward_smoke(cfg),
    }
