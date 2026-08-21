"""Optional LTX / long-form sports integration plan."""

from __future__ import annotations

from typing import Any


def ltx_integration_plan() -> dict[str, Any]:
    return {
        "summary": "Use TF-SELECTOR segment saliency as fold sidecars for hour-long sports prep",
        "fold_keys": ["svhighlights_saliency", "svhighlights_segment_score"],
        "sidecar_fields": {
            "clip_saliency": "0–5 TF-SELECTOR score per 2 s clip",
            "segment_id": "context-aware segment index",
            "highlight_overlap": "fraction overlap with aligned official highlight labels",
        },
        "training_hooks": [
            "Weight AV-fold clips by svhighlights_saliency during merged_native sports expansion",
            "Filter precompute to top-K salient segments before VAE encode on GPU0",
        ],
        "inference_hooks": [
            "GOPEX_INFER_SVHIGHLIGHTS=1 — trim delivery to top salient spans for recap mode",
            "Autocut fold bridge: segment scores → RQ-VAE edit candidates",
        ],
        "scripts": ["./scripts/gopex-svhighlights.sh", "./scripts/kino-expansion-new-datasets.sh"],
        "env": {
            "GOPEX_SVHIGHLIGHTS_ENABLE": "1",
            "GOPEX_INFER_SVHIGHLIGHTS": "1",
            "GOPEX_SVHIGHLIGHTS_TOP_FRACTION": "0.15",
            "GOPEX_SVHIGHLIGHTS_VLM": "InternVL2.5-8B",
            "GOPEX_SVHIGHLIGHTS_LLM": "Llama-3-8B",
        },
    }


def gopex_env_snippet() -> str:
    plan = ltx_integration_plan()
    lines = [f'export {k}="{v}"' for k, v in plan["env"].items()]
    return "\n".join(lines)
