"""SSL art classification framework card and WikiArt Table 1 (IRCDL'26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ssl_art.classify import (
    accuracy,
    knn_zero_shot_predict,
    linear_predict,
    zero_shot_predict,
)
from ltx_trainer.ssl_art.config import SSLArtConfig
from ltx_trainer.ssl_art.layout import LIMITATIONS
from ltx_trainer.ssl_art.prompts import build_label_prompts
from ltx_trainer.ssl_art.retrieval import top_k_retrieve


def framework_card(cfg: SSLArtConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SSLArtConfig()
    return {
        "name": "Harnessing Self-Supervised Features for Art Classification",
        "venue": cfg.venue,
        "authors": "Melis, Bilardello, Prato, Turri, Baraldi (Unimore)",
        "idea": (
            "Frozen DINOv3 / CLIP vision encoders + decoupled heads: zero-shot, "
            "KNN zero-shot, or linear classification on WikiArt style (27) and genre (10)."
        ),
        "dataset": {
            "name": cfg.dataset,
            "paintings": cfg.dataset_size,
            "style_classes": cfg.style_classes,
            "genre_classes": cfg.genre_classes,
            "splits": f"{cfg.train_split:.0%}/{cfg.val_split:.0%}/{cfg.test_split:.0%}",
        },
        "backbones": {
            "supervised": cfg.supervised_baseline,
            "ssl_clip": cfg.clip_backbone,
            "ssl_dino": cfg.dino_backbone,
        },
        "strategies": ["zero_shot", "knn_zero_shot", "linear"],
        "applications": ["VR/AR museum navigation", "curator labeling", "visual retrieval"],
        "defaults": cfg.__dict__,
    }


def table_i_wikiart_results() -> list[dict[str, Any]]:
    """Table 1 — WikiArt test-set style/genre metrics (paper)."""
    return [
        {
            "model": "EfficientNetV2",
            "trainable": True,
            "style": {"P": 68.9, "R": 68.6, "F1": 68.3, "acc@1": 68.6},
            "genre": {"P": 82.0, "R": 82.2, "F1": 82.0, "acc@1": 82.2},
        },
        {
            "model": "DINO Zero-Shot",
            "trainable": False,
            "style": {"P": 29.1, "R": 34.8, "F1": 26.1, "acc@1": 27.4},
            "genre": {"P": 66.9, "R": 67.5, "F1": 63.7, "acc@1": 67.2},
        },
        {
            "model": "CLIP Zero-Shot",
            "trainable": False,
            "style": {"P": 39.3, "R": 41.2, "F1": 36.8, "acc@1": 41.9},
            "genre": {"P": 69.4, "R": 65.6, "F1": 60.2, "acc@1": 64.4},
        },
        {
            "model": "DINO-KNN",
            "trainable": False,
            "style": {"P": 63.5, "R": 62.4, "F1": 61.7, "acc@1": 63.2},
            "genre": {"P": 78.5, "R": 78.0, "F1": 78.1, "acc@1": 80.2},
        },
        {
            "model": "CLIP-KNN",
            "trainable": False,
            "style": {"P": 69.2, "R": 68.6, "F1": 68.3, "acc@1": 70.6},
            "genre": {"P": 80.3, "R": 80.9, "F1": 80.5, "acc@1": 81.7},
        },
        {
            "model": "DINO-Linear",
            "trainable": True,
            "style": {"P": 71.5, "R": 61.4, "F1": 65.0, "acc@1": 65.0},
            "genre": {"P": 81.6, "R": 80.6, "F1": 81.0, "acc@1": 83.6},
        },
        {
            "model": "CLIP-Linear (Ours)",
            "trainable": True,
            "style": {"P": 74.3, "R": 67.2, "F1": 69.8, "acc@1": 69.2},
            "genre": {"P": 83.2, "R": 82.5, "F1": 82.8, "acc@1": 84.9},
        },
    ]


def best_clip_linear_metrics() -> dict[str, float]:
    row = next(r for r in table_i_wikiart_results() if r["model"] == "CLIP-Linear (Ours)")
    return {"style_acc@1": row["style"]["acc@1"], "genre_acc@1": row["genre"]["acc@1"]}


def pipeline_demo(cfg: SSLArtConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    """Toy smoke: three strategies on synthetic embeddings."""
    cfg = cfg or SSLArtConfig()
    rng = np.random.default_rng(seed)
    dim = 16
    labels = ["Impressionism", "Baroque", "Cubism"]
    n_ref = 30
    ref_feats = rng.standard_normal((n_ref, dim))
    ref_labels = [labels[i % len(labels)] for i in range(n_ref)]
    query_feat = ref_feats[0] + 0.05 * rng.standard_normal(dim)

    text_feats = rng.standard_normal((len(labels), dim))
    text_feats[0] = query_feat + 0.1 * rng.standard_normal(dim)

    zs = zero_shot_predict(query_feat, text_feats, labels)
    knn = knn_zero_shot_predict(query_feat, ref_feats, ref_labels, k=cfg.knn_k)
    w = rng.standard_normal((len(labels), dim)) * 0.1
    b = np.zeros(len(labels))
    lin = linear_predict(query_feat, w, b, labels)

    gallery_ids = [f"wikiart_{i}" for i in range(n_ref)]
    retrieved = top_k_retrieve(query_feat, ref_feats, gallery_ids, k=5)

    gold = [labels[0]] * 3
    return {
        "zero_shot_pred": zs,
        "knn_pred": knn,
        "linear_pred": lin,
        "style_prompts": build_label_prompts(labels[:2], task="style"),
        "genre_prompts": build_label_prompts(["landscape", "portrait"], task="genre"),
        "retrieval_top5": retrieved,
        "toy_batch_acc": accuracy([zs, knn, lin], gold),
    }


def evaluation_demo(cfg: SSLArtConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["best_paper_metrics"] = best_clip_linear_metrics()
    demo["limitations"] = LIMITATIONS
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_wikiart": table_i_wikiart_results(),
        "best_method": "CLIP-Linear (Ours)",
        "headline": {
            "clip_linear_beats_efficientnet_style_acc": True,
            "clip_knn_no_train_style_acc": 70.6,
            "efficientnet_style_acc": 68.6,
        },
        "implementation_notes": {
            "zero_shot_dino_text": "dino.txt encoder (CVPR 2025)",
            "knn_reference": "WikiArt train split, query = test",
            "retrieval_index": "FAISS cosine on frozen CLS features",
        },
    }
