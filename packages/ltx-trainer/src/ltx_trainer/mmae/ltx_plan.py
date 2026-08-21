"""LTX integration plan for MMAE-style audio editing eval sidecars."""

from __future__ import annotations

from typing import Any


def ltx_integration_plan() -> dict[str, Any]:
    return {
        "goal": "Tag AV-fold clips with MMAE taxonomy + rubric metadata for instruction-editing QA",
        "fold_hook": "mmae",
        "audio_fold_hook": "mmae_audio",
        "sidecar_fields": [
            "mmae.modality",
            "mmae.complexity",
            "mmae.granularity",
            "mmae.operations",
            "mmae.rubric_count",
            "mmae.instruction_preview",
        ],
        "eval_harness": "tools/mmae_run_eval.py + gopex-mmae.sh",
        "judger_modes": ["mock", "openai", "omni"],
        "local_baselines": ["Identity", "Noise"],
        "dataset": "BoJack/MMAE on Hugging Face",
        "bootstrap": [
            "./scripts/gopex-mmae.sh bootstrap-dev --root \"$GOPEX_MMAE_ROOT\"",
            "./scripts/gopex-mmae.sh bootstrap-hub --root \"$GOPEX_MMAE_ROOT\" --limit 100",
            "./scripts/gopex-mmae.sh baseline-identity /tmp/mmae_preds --root \"$GOPEX_MMAE_ROOT\"",
            "./scripts/gopex-mmae.sh eval-submission /tmp/mmae_preds --root \"$GOPEX_MMAE_ROOT\" --model Identity",
        ],
        "hub_sync": "./scripts/gopex-mmae.sh sync-hub --root \"$GOPEX_MMAE_ROOT\" --limit 100",
        "hub_audio": "./scripts/gopex-mmae.sh sync-hub-audio --root \"$GOPEX_MMAE_ROOT\" --limit 100",
        "editor_plan": "./scripts/gopex-mmae.sh editor-plan --model Step-Audio-EditX --predictions /tmp/mmae_preds --root \"$GOPEX_MMAE_ROOT\"",
        "predictions_json": "./scripts/gopex-mmae.sh build-predictions /tmp/mmae_preds /tmp/mmae_preds.json --root \"$GOPEX_MMAE_ROOT\"",
    }


def gopex_env_snippet() -> str:
    return "\n".join(
        [
            "export GOPEX_MMAE_ENABLE=1",
            'export GOPEX_MMAE_ROOT="${GOPEX_MMAE_ROOT:-$GOPEX_OPEN_DATA_ROOT/mmae}"',
            "export GOPEX_MMAE_JUDGER=qwen3-omni",
            "export GOPEX_MMAE_OMNI_BASE_URL=${QWEN_BASE_URL:-http://127.0.0.1:8011/v1}",
        ]
    )
