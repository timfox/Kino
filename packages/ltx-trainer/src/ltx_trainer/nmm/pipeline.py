"""NMM Roadmap framework card, paper tables, and smoke demos (arXiv:2605.25343)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nmm.config import NMMConfig
from ltx_trainer.nmm.nativity import FusionRegime, describe_fusion
from ltx_trainer.nmm.taxonomy import IOCategory, describe_io


def framework_card(cfg: NMMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NMMConfig()
    return {
        "name": "NMM Roadmap",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "scope": "Native multimodal modeling — architecture, data, training, inference, evaluation",
        "fusion_regimes": [r.value for r in FusionRegime],
        "io_paradigms": [c.value for c in IOCategory],
        "lifecycle_sections": ["architecture", "dataset", "training", "inference", "evaluation"],
    }


def table_native_models() -> list[dict[str, Any]]:
    """Table 1 excerpt — flagship open NMM systems (modal I/O flags)."""
    # Text/Img/Aud/Vid for in and out per paper Table 1
    rows = [
        {
            "name": "Qwen3-VL",
            "date": "2025.09",
            "params": "235B-A22B",
            "io": "M2T",
            "fusion": "mid-fusion",
            "in": "TIV",
            "out": "T",
        },
        {
            "name": "Kimi K2.5",
            "date": "2026.01",
            "params": "1T-A32B",
            "io": "M2T",
            "fusion": "mid-fusion",
            "in": "TI",
            "out": "T",
        },
        {
            "name": "LTX-2.3",
            "date": "2026.03",
            "params": "19B",
            "io": "M2G",
            "fusion": "mid-fusion",
            "in": "TIVA",
            "out": "VA",
        },
        {
            "name": "Qwen3-Omni",
            "date": "2025.09",
            "params": "30B-A3B",
            "io": "M2G",
            "fusion": "mid-fusion",
            "in": "TIVA",
            "out": "TA",
        },
        {
            "name": "BAGEL",
            "date": "2025.05",
            "params": "14B-A7B",
            "io": "M2M",
            "fusion": "mid-fusion",
            "in": "TI",
            "out": "TI",
        },
        {
            "name": "Emu3.5",
            "date": "2025.10",
            "params": "34.1B",
            "io": "M2M",
            "fusion": "early-fusion",
            "in": "TI",
            "out": "TIV",
        },
        {
            "name": "Transfusion",
            "date": "2024.08",
            "params": "7B",
            "io": "M2M",
            "fusion": "early-fusion",
            "in": "TI",
            "out": "TI",
        },
        {
            "name": "Chameleon",
            "date": "2024.05",
            "params": "34B",
            "io": "M2M",
            "fusion": "early-fusion",
            "in": "TI",
            "out": "TI",
        },
        {
            "name": "Moshi",
            "date": "2024.09",
            "params": "7B",
            "io": "M2M",
            "fusion": "early-fusion",
            "in": "TA",
            "out": "TA",
        },
        {
            "name": "MiniCPM-o-4.5",
            "date": "2026.02",
            "params": "9B",
            "io": "M2G",
            "fusion": "mid-fusion",
            "in": "TIVA",
            "out": "TA",
        },
    ]
    return rows


def table_training_data() -> dict[str, list[str]]:
    """Table 2 — NMM training data by functional role (§4)."""
    return {
        "understand": [
            "LAION-5B",
            "VQA v2",
            "MMC4",
            "DocVQA",
            "RefCOCO",
            "MSR-VTT",
            "AudioSet",
        ],
        "generate": [
            "DiffusionDB",
            "InstructPix2Pix",
            "WebVid-10M",
            "LibriTTS",
            "AudioCaps",
        ],
        "interact": [
            "Mind2Web",
            "WebArena",
            "Android-in-the-Wild",
            "OSWorld",
            "ALFWorld",
        ],
        "align": [
            "LLaVA-RLHF",
            "VLFeedback",
            "ImageReward",
            "VBench",
            "SPA-VL",
        ],
    }


def table_evaluation_benchmarks() -> dict[str, list[dict[str, str]]]:
    """Table 3 excerpt — evaluation benchmarks by modality (§7)."""
    return {
        "image": [
            {"benchmark": "MMBench", "metric": "Acc.", "task": "general perception"},
            {"benchmark": "MMMU", "metric": "Acc.", "task": "knowledge reasoning"},
            {"benchmark": "POPE", "metric": "F1", "task": "hallucination"},
            {"benchmark": "GenEval", "metric": "Comp. Score", "task": "T2I composition"},
        ],
        "audio": [
            {"benchmark": "LibriSpeech", "metric": "WER", "task": "ASR"},
            {"benchmark": "Moshi Eval", "metric": "Latency", "task": "full-duplex"},
            {"benchmark": "Full-Duplex-Bench", "metric": "Multi", "task": "turn-taking"},
        ],
        "video": [
            {"benchmark": "VideoMME", "metric": "Acc.", "task": "offline QA"},
            {"benchmark": "OVO-Bench", "metric": "Multi", "task": "streaming perception"},
            {"benchmark": "VBench", "metric": "Multi", "task": "generation quality"},
            {"benchmark": "SeedVideoBench 2.0", "metric": "6-dim", "task": "A/V sync generation"},
        ],
    }


def classify_model(name: str) -> dict[str, Any] | None:
    """Lookup IO + fusion tags for a Table 1 model name."""
    for row in table_native_models():
        if row["name"].lower() == name.lower():
            return {
                **row,
                "io_detail": describe_io(IOCategory[row["io"]]),
                "fusion_detail": describe_fusion(FusionRegime(row["fusion"])),
            }
    return None


def roadmap_demo(cfg: NMMConfig | None = None) -> dict[str, Any]:
    """Smoke: nativity ordering + Table 1 category counts."""
    cfg = cfg or NMMConfig()
    models = table_native_models()
    by_io: dict[str, int] = {}
    by_fusion: dict[str, int] = {}
    for m in models:
        by_io[m["io"]] = by_io.get(m["io"], 0) + 1
        by_fusion[m["fusion"]] = by_fusion.get(m["fusion"], 0) + 1

    ltx = classify_model("LTX-2.3")
    assert ltx is not None

    return {
        "paper": cfg.paper_arxiv,
        "native_model_count": len(models),
        "by_io": by_io,
        "by_fusion": by_fusion,
        "early_fusion_models": [m["name"] for m in models if m["fusion"] == "early-fusion"],
        "ltx23": {
            "io": ltx["io"],
            "fusion": ltx["fusion"],
            "outputs": ltx["out"],
        },
        "convergence_axis": "M2T/M2G → symmetric M2M (§8.1)",
    }


def evaluation_demo(cfg: NMMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NMMConfig()
    demo = roadmap_demo(cfg)
    data = table_training_data()
    bench = table_evaluation_benchmarks()

    return {
        **demo,
        "training_data_categories": len(data),
        "total_training_datasets_listed": sum(len(v) for v in data.values()),
        "eval_modalities": list(bench.keys()),
        "has_streaming_video_bench": any(
            b["benchmark"] == "OVO-Bench" for b in bench["video"]
        ),
        "m2m_count_ge_m2t": demo["by_io"].get("M2M", 0) >= demo["by_io"].get("M2T", 0)
        or demo["by_io"].get("M2M", 0) > 0,
    }
