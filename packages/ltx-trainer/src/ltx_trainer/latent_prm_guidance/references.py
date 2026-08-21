"""Key citation anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, Any]]:
    return [
        {"id": "hao2025", "cite": "Hao et al. 2025", "topic": "COCONUT continuous latent reasoning"},
        {"id": "zou2025", "cite": "Zou et al. 2025", "topic": "Training-free alignment transform Wa"},
        {"id": "bitan2025", "cite": "Bitan et al. 2025", "topic": "UniPar / ParaTrans parallel translation"},
        {"id": "li2025", "cite": "Li et al. 2025", "topic": "CodePRM post-decode process supervision"},
        {"id": "cobbe2021", "cite": "Cobbe et al. 2021", "topic": "Verifier-guided reasoning template"},
    ]
