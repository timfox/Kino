"""ExtrAnom framework card, paper tables, and smoke demos (arXiv:2605.25806)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.extranom.categories import (
    anomaly_category_counts,
    category_share_of_total,
    table_category_statistics,
)
from ltx_trainer.extranom.config import ExtrAnomConfig
from ltx_trainer.extranom.metrics import caption_metrics


def framework_card(cfg: ExtrAnomConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ExtrAnomConfig()
    counts = anomaly_category_counts()
    return {
        "name": "ExtrAnom",
        "paper": cfg.paper_arxiv,
        "title": "Women-centric multi-modal VAD benchmark",
        "repo": cfg.repo_url,
        "total_videos": cfg.total_videos,
        "normal": cfg.normal_videos,
        "anomalous": cfg.anomalous_videos,
        "anomaly_classes": list(k for k in counts if k != "normal"),
        "textual_annotations_per_video": cfg.textual_annotations_per_video,
        "train_test": f"{cfg.train_videos}/{cfg.test_videos} ({cfg.train_ratio:.0%} split)",
        "focus": "chain snatching, stalking, harassment, kidnapping, assassination",
        "conditions": "low-light, low-resolution, long-shot surveillance footage",
    }


def table_dataset_comparison() -> list[dict[str, Any]]:
    """Table 1 — ExtrAnom vs existing VAD datasets."""
    rows = [
        ("UCF-Crime", 1900, 13, "partial", False),
        ("UCA", 1854, 13, "partial", True),
        ("UBnormal", 543, 22, "no", False),
        ("XD-Violence", 4754, 6, "partial", False),
        ("RareAnom", 2200, 17, "partial", False),
        ("ExtrAnom (Ours)", 1001, 5, "yes", True),
    ]
    return [
        {
            "dataset": name,
            "num_videos": n,
            "anomaly_classes": c,
            "women_centric": wc,
            "textual_annotation": txt,
        }
        for name, n, c, wc, txt in rows
    ]


def table_vlm_caption_similarity() -> dict[str, dict[str, float]]:
    """Table 3 — video-level description similarity vs ground truth."""
    return {
        "Video-ChatGPT": {"bleu": 0.0163, "bert": 0.2627, "cider": 0.2082, "meteor": 0.1818, "rouge": 0.2411},
        "Video-LLaMA": {"bleu": 0.0138, "bert": 0.1947, "cider": 0.1639, "meteor": 0.1625, "rouge": 0.2069},
        "Video-LLaVA": {"bleu": 0.0059, "bert": 0.1409, "cider": 0.1159, "meteor": 0.1367, "rouge": 0.1374},
        "LLaVA-Next-Video": {"bleu": 0.0056, "bert": 0.0192, "cider": 0.1271, "meteor": 0.0648, "rouge": 0.1120},
        "QwenVL3": {"bleu": 0.0118, "bert": 0.2233, "cider": 0.0009, "meteor": 0.2376, "rouge": 0.2014},
        "InternVL3": {"bleu": 0.0061, "bert": 0.2100, "cider": 0.0004, "meteor": 0.2116, "rouge": 0.1439},
        "Gemini": {"bleu": 0.0197, "bert": 0.2946, "cider": 0.0866, "meteor": 0.2572, "rouge": 0.2739},
        "Holmes-VAU": {"bleu": 0.0201, "bert": 0.1956, "cider": 0.2397, "meteor": 0.1423, "rouge": 0.2243},
        "Holmes-VAU (fine-tuned on ExtrAnom)": {
            "bleu": 0.0824,
            "bert": 0.3270,
            "cider": 0.3820,
            "meteor": 0.1743,
            "rouge": 0.3850,
        },
        "LAVAD": {"bleu": 0.0058, "bert": 0.1271, "cider": 0.0720, "meteor": 0.1314, "rouge": 0.1327},
    }


def table_vision_auc() -> dict[str, dict[str, float | None]]:
    """Table 4 — AUC % on ExtrAnom (inference only vs trained on ExtrAnom)."""
    return {
        "MIL": {"feature": "C3D", "inference_extranom": 49.62, "trained_extranom": 55.63},
        "BODS": {"feature": "I3D", "inference_extranom": 39.25, "trained_extranom": 49.52},
        "GODS": {"feature": "I3D", "inference_extranom": 40.63, "trained_extranom": 53.72},
        "RTFM": {"feature": "I3D", "inference_extranom": 44.56, "trained_extranom": 65.39},
        "MIST": {"feature": "I3D", "inference_extranom": 41.60, "trained_extranom": 57.25},
        "RareAnom": {"feature": "I3D", "inference_extranom": 40.89, "trained_extranom": 55.47},
        "DyAnNet": {"feature": "I3D", "inference_extranom": 53.25, "trained_extranom": 67.53},
        "CLIP-TSA": {"feature": "ViT", "inference_extranom": 51.00, "trained_extranom": 61.20},
        "VadCLIP": {"feature": "ViT", "inference_extranom": 50.60, "trained_extranom": 70.89},
    }


def holmes_vau_error_analysis() -> dict[str, dict[str, float]]:
    """Fig. 2 / Sec. 1 — pretrained Holmes-VAU errors on UCF-Crime vs ExtrAnom."""
    return {
        "misclassified_as_normal_pct": {"extranom": 65.78, "ucf_crime": 15.48},
        "incorrect_description_women_violence_pct": {"extranom": 62.45, "ucf_crime": 37.80},
        "gender_misclassification_pct": {"extranom": 66.8, "ucf_crime": 64.9},
        "dangerous_object_misclassification_pct": {"extranom": 77.78, "ucf_crime": 58.33},
    }


def example_ground_truth_chain_snatching() -> str:
    """Fig. 6 ground truth for Chain_Snatching_v163."""
    return (
        "A woman dressed in a black saree and a pink blouse is approached from behind "
        "by a man wearing a white shirt and black pants. The man snatches the chain from "
        "her neck and runs away, and the woman starts running after him."
    )


def training_step_demo(cfg: ExtrAnomConfig | None = None) -> dict[str, float]:
    """Smoke: caption metrics on ground truth vs a generic VLM-style description."""
    cfg = cfg or ExtrAnomConfig()
    gt = example_ground_truth_chain_snatching()
    bad = (
        "The video shows normal traffic flow with a person walking near parked cars "
        "in daylight surveillance footage with no unusual activity."
    )
    good = (
        "A woman in a black saree is approached from behind; a man snatches her chain "
        "and flees while she chases him."
    )
    m_bad = caption_metrics(bad, gt)
    m_good = caption_metrics(good, gt)
    return {
        "bad_token_f1": m_bad["token_f1"],
        "good_token_f1": m_good["token_f1"],
        "good_beats_bad": float(m_good["token_f1"] > m_bad["token_f1"]),
        "train_videos": float(cfg.train_videos),
        "annotations_per_video": float(cfg.textual_annotations_per_video),
    }


def evaluation_demo(cfg: ExtrAnomConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ExtrAnomConfig()
    step = training_step_demo(cfg)
    vlm = table_vlm_caption_similarity()
    auc = table_vision_auc()
    holmes = holmes_vau_error_analysis()

    ft = vlm["Holmes-VAU (fine-tuned on ExtrAnom)"]
    pre = vlm["Holmes-VAU"]
    vadclip = auc["VadCLIP"]

    return {
        **step,
        "stalking_share_pct": category_share_of_total()["stalking"],
        "holmes_ft_bleu_gain": ft["bleu"] - pre["bleu"],
        "holmes_ft_rouge_gain": ft["rouge"] - pre["rouge"],
        "vadclip_auc_inference": vadclip["inference_extranom"],
        "vadclip_auc_trained": vadclip["trained_extranom"],
        "training_improves_auc": float(vadclip["trained_extranom"] > vadclip["inference_extranom"]),
        "extranom_gender_error_pct": holmes["gender_misclassification_pct"]["extranom"],
        "only_women_centric_five_class": True,
        "category_stats": table_category_statistics(),
    }
