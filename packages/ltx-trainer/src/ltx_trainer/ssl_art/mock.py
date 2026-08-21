"""SSL art CLIP/DINO classify smoke (IRCDL'26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ssl_art.classify import knn_zero_shot_predict, zero_shot_predict
from ltx_trainer.ssl_art.config import SSLArtConfig


def evaluation_smoke(cfg: SSLArtConfig | None = None) -> dict[str, Any]:
    c = cfg or SSLArtConfig()
    rng = np.random.default_rng(0)
    img = rng.standard_normal(16)
    text = rng.standard_normal((3, 16))
    labels = ["impressionism", "baroque", "renaissance"]
    pred = zero_shot_predict(img, text, labels)
    knn = knn_zero_shot_predict(img, text, labels, k=2)
    return {
        "paper": "IRCDL'26 WikiArt SSL art",
        "zero_shot_label": pred,
        "knn_label": knn,
    }
