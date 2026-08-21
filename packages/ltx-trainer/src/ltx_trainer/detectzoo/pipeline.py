"""Framework card, tables, evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.detectzoo.api import load_detector
from ltx_trainer.detectzoo.config import DetectZooConfig
from ltx_trainer.detectzoo.evaluator import BenchmarkEvaluator, DatasetItem
from ltx_trainer.detectzoo.metrics import toolkit_comparison_table
from ltx_trainer.detectzoo.registry import ALL_DETECTORS, DATASETS


def framework_card(cfg: DetectZooConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DetectZooConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "github": cfg.github,
        "pypi": cfg.pypi,
        "n_detectors": cfg.n_detectors,
        "n_datasets": cfg.n_datasets,
        "modalities": {
            "text": cfg.n_text_detectors,
            "image": cfg.n_image_detectors,
            "audio": cfg.n_audio_detectors,
        },
        "api": "load_detector(name) -> predict() -> DetectionResult",
        "metrics": ["AUROC", "AUPR", "AP", "EER", "accuracy", "F1", "TPR", "FPR"],
        "principles": ["reproducibility", "accessibility", "extensibility"],
    }


def detector_counts() -> dict[str, int]:
    return {k: len(v) for k, v in ALL_DETECTORS.items()}


def dataset_counts() -> dict[str, int]:
    return {k: len(v) for k, v in DATASETS.items()}


def headline_findings() -> dict[str, Any]:
    """Section 4.2 summary anchors."""
    return {
        "text": "Task semantics dominate; rewrite/polish hardest; GPT-4o/Qwen hardest sources",
        "image": "CLIP/hybrid (FatFormer, AIDE, SAFE) generalize best cross-architecture",
        "audio": "Large SSL AntiDeepfake models strong OOD on ASVspoof/FoR/In-the-Wild",
    }


def forward_smoke(cfg: DetectZooConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DetectZooConfig()
    text_det = load_detector("fast_detectgpt", cfg=cfg)
    image_det = load_detector("aeroblade", cfg=cfg)
    audio_det = load_detector("rawnet2", cfg=cfg)
    text_r = text_det.predict("This passage exhibits uniform token statistics typical of LLM output.")
    image_r = image_det.predict("/tmp/synthetic.png")
    audio_r = audio_det.predict("/tmp/clip.wav")
    items = [
        DatasetItem("Human-written editorial with irregular rhythm.", 0),
        DatasetItem("Highly uniform synthetic summary with predictable cadence.", 1),
        DatasetItem("Another human sample with varied vocabulary choices.", 0),
        DatasetItem("Machine generated text with low perplexity curvature.", 1),
    ]
    ev = BenchmarkEvaluator(
        dataset_name="demo_text",
        items=items,
        detectors=[text_det],
    )
    return {
        "text": text_r.to_dict(),
        "image": image_r.to_dict(),
        "audio": audio_r.to_dict(),
        "benchmark": ev.run(),
    }


def table1_toolkit_comparison() -> list[dict[str, Any]]:
    return toolkit_comparison_table()


def table17_asvspoof_dedicated() -> list[dict[str, Any]]:
    """Table 17 — ASVspoof 2019 LA subset (Ge et al. reproduction setting)."""
    return [
        {"detector": "aasist", "eer": 0.01, "auroc": 0.9992, "f1": 0.9755},
        {"detector": "rawgat_st", "eer": 0.006, "auroc": 0.9984, "f1": 0.9869},
        {"detector": "rawnet2", "eer": 0.052, "auroc": 0.9876, "f1": 0.9505},
        {"detector": "res_tssdnet", "eer": 0.012, "auroc": 0.9995, "f1": 0.9900},
        {"detector": "samo", "eer": 0.052, "auroc": 0.9780, "f1": 0.9041},
        {"detector": "ast_asvspoof", "eer": 0.062, "auroc": 0.9814, "f1": 0.9370},
    ]


def table18_antideepfake_cross_dataset() -> list[dict[str, Any]]:
    """Table 18 — AntiDeepfake + XLS-R SLS cross-dataset EER (%)."""
    return [
        {"detector": "anti_deepfake_wav2vec", "asvspoof2019": 0.2, "for": 7.2, "in_the_wild": 1.6},
        {"detector": "anti_deepfake_hubert", "asvspoof2019": 0.0, "for": 11.8, "in_the_wild": 4.8},
        {"detector": "anti_deepfake_xlsr2b", "asvspoof2019": 0.6, "for": 8.8, "in_the_wild": 1.2},
        {"detector": "xlsr_sls", "asvspoof2019": 0.4, "for": 11.8, "in_the_wild": 12.8},
    ]


def evaluation_demo(cfg: DetectZooConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DetectZooConfig()
    return {
        "framework": framework_card(cfg),
        "toolkit_table1": toolkit_comparison_table(),
        "detector_counts": detector_counts(),
        "dataset_counts": dataset_counts(),
        "headline_findings": headline_findings(),
        "anchors": {
            "fast_detectgpt_writingprompts_auroc": cfg.fast_detectgpt_wp_auroc,
            "res_tssdnet_asvspoof_eer": cfg.asvspoof_res_tssdnet_eer,
        },
        "forward": forward_smoke(cfg),
    }
