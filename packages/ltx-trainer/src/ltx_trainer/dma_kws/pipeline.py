"""Framework card and paper benchmark excerpts for DMA-KWS (arXiv:2605.22120)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dma_kws.config import DmaKwsConfig
from ltx_trainer.dma_kws.layout import LIMITATIONS
from ltx_trainer.dma_kws.mock import evaluation_smoke


def framework_card(cfg: DmaKwsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DmaKwsConfig()
    return {
        "name": "DMA-KWS — effective user-defined keyword spotting",
        "paper": cfg.paper_arxiv,
        "thesis": "Dual-stage (CTC search → QbyT phoneme matcher) + multimodal enrollment + lightweight continual adaptation.",
        "stages": {
            "stage1": "CTC decoding with streaming phoneme search to localize candidate segments (Score1).",
            "stage2": "QbyT phoneme matcher to verify candidates (Score2), improving confusable discrimination.",
        },
        "enrollment": {
            "speaker_independent": "text-only phoneme enrollment (QbyT).",
            "speaker_dependent": "multi-modal enrollment fusing text + enrolled audio via MAM.",
        },
        "adaptation": {
            "mechanism": "parameter-efficient LoRA updates on matcher attention (QKV) using synthetic + real feedback.",
            "updated_params": cfg.lora_updated_params_total,
        },
        "headline": {
            "LibriPhrase_H_auc": cfg.headline_auc_lph,
            "LibriPhrase_H_eer": cfg.headline_eer_lph,
        },
        "limitations": LIMITATIONS,
    }


def table_i_si_kws_main() -> list[dict[str, Any]]:
    """Table I excerpt: best DMA-KWS (2) row (LPH/LPE + out-of-domain)."""
    return [
        {
            "method": "DMA-KWS(2)",
            "params_m": 4.1,
            "pretrain": "1460h†",
            "finetune": "1460h",
            "auroc": {"LPH": 97.85, "LPE": 99.98, "GSC": 99.21, "Qcomm": 99.90},
            "eer": {"LPH": 6.13, "LPE": 0.45, "GSC": 3.93, "Qcomm": 1.52},
        }
    ]


def table_ii_dual_scaling() -> list[dict[str, Any]]:
    """Table II excerpt: dual data scaling (P-WER + AUROC/EER)."""
    return [
        {
            "setting": "Stage1 LS-460; Stage2 LP-460",
            "dma1_pwer": {"LSclean": 4.44, "LSother": 13.39},
            "dma1_lph": {"auroc": 95.33, "eer": 10.78},
            "dma2_lph": {"auroc": 97.03, "eer": 7.97},
        },
        {
            "setting": "Stage1 LS-GS-1460; Stage2 LP-GP-1460",
            "dma1_pwer": {"LSclean": 4.45, "LSother": 11.80},
            "dma1_lph": {"auroc": 95.77, "eer": 10.02},
            "dma2_lph": {"auroc": 97.85, "eer": 6.13},
        },
    ]


def table_iv_sd_kws() -> list[dict[str, Any]]:
    """Table IV excerpt: best DMA-KWS(4)@AT row (SD-KWS)."""
    return [
        {
            "method": "DMA-KWS(4)@AT",
            "infer_params_m": 4.1,
            "enroll_params_m": 3.6,
            "auroc": {"LPSD_H": 97.70, "LPSD_E": 99.98, "QcommSD": 99.97, "AudioMNIST": 99.80},
            "eer": {"LPSD_H": 6.58, "LPSD_E": 0.31, "QcommSD": 0.88, "AudioMNIST": 1.67},
        }
    ]


def table_v_hey_snips_zero_shot() -> list[dict[str, Any]]:
    """Table V excerpt: Hey-Snips recall@FARs for DMA-KWS variants."""
    return [
        {"method": "CTC-Streaming", "recall@far_0.05": 98.06, "recall@far_0.5": 98.89, "recall@far_1.0": 98.97},
        {"method": "DMA-KWS(2)", "recall@far_0.05": 99.45, "recall@far_0.5": 99.76, "recall@far_1.0": 99.80},
        {"method": "DMA-KWS(4)", "recall@far_0.05": 99.72, "recall@far_0.5": 99.84, "recall@far_1.0": 99.84},
    ]


def table_xii_inference_time() -> list[dict[str, Any]]:
    """Table XII excerpt: total inference time on LibriSpeech test (10.75h)."""
    return [
        {"method": "CTC-Streaming", "rtx4090_s": 463, "i9_s": 594, "m2_s": 759},
        {"method": "DMA-KWS(1)", "rtx4090_s": 466, "i9_s": 604, "m2_s": 769},
        {"method": "DMA-KWS(2)", "rtx4090_s": 481, "i9_s": 664, "m2_s": 847},
    ]


def headline_results() -> dict[str, Any]:
    cfg = DmaKwsConfig()
    return {
        "LPH_auc": cfg.headline_auc_lph,
        "LPH_eer": cfg.headline_eer_lph,
        "lora_updated_params_total": cfg.lora_updated_params_total,
    }


def evaluation_demo(cfg: DmaKwsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DmaKwsConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_si_kws_main": table_i_si_kws_main(),
        "table_ii_dual_scaling": table_ii_dual_scaling(),
        "table_iv_sd_kws": table_iv_sd_kws(),
        "table_v_hey_snips_zero_shot": table_v_hey_snips_zero_shot(),
        "table_xii_inference_time": table_xii_inference_time(),
        "headlines": headline_results(),
    }

