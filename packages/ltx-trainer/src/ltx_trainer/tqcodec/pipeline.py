"""Pipeline and agent knowledge."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tqcodec.config import (
    BITRATES_KBPS,
    DECODER_GMACS,
    LOSS_WEIGHTS,
    PAPER_DOI,
    PAPER_TITLE,
    PAPER_URL,
    RECEPTIVE_FIELD_SAMPLES,
)
from ltx_trainer.tqcodec.metrics import (
    table1_codec_comparison,
    table3_ablation,
    table4_subband_baseline,
    table5_subjective_mos,
)
from ltx_trainer.tqcodec.synthetic import dataset_summary


def knowledge() -> dict[str, Any]:
    return {
        "name": "TQCodec",
        "paper": PAPER_TITLE,
        "authors": ["Lixing He", "Zhouxuan Chen", "Mingshuai Liu", "et al."],
        "affiliation": "Tencent Music Entertainment",
        "url": PAPER_URL,
        "doi": PAPER_DOI,
        "sample_rate_khz": 44.1,
        "bitrates_kbps": list(BITRATES_KBPS),
        "decoder_gmacs": DECODER_GMACS,
        "receptive_field_samples": RECEPTIVE_FIELD_SAMPLES,
        "innovations": ["SEANet", "RSimVQ", "waveform loss", "PQMF subband allocation"],
    }


def paper_report() -> dict[str, Any]:
    return {
        "table1_codecs": table1_codec_comparison(),
        "table2_datasets": dataset_summary()["datasets"],
        "table3_ablation": table3_ablation(),
        "table4_subband": table4_subband_baseline(),
        "table5_subjective": table5_subjective_mos(),
        "loss_weights": LOSS_WEIGHTS,
    }
