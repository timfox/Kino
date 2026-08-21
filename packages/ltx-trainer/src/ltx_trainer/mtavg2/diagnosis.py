"""LTX native prep and clip-level cinematic failure diagnosis hooks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ltx_trainer.mtavg2.benchmarks import FIG4_FAILURE_RATE_PCT, TABLE4_HOLISTIC_METRICS
from ltx_trainer.mtavg2.prompts import LTX_SCRIPT_PROMPT_SUFFIX
from ltx_trainer.mtavg2.scoring import holistic_quality_index
from ltx_trainer.mtavg2.taxonomy import FAILURE_MODE_BY_ID, SUB_DIMENSION_BY_CODE


def mtavg2_diag_enabled() -> bool:
    return os.environ.get("GOPEX_MTAVG2_DIAG", "0").strip().lower() in ("1", "true", "yes")


def mtavg2_fold_enabled() -> bool:
    raw = os.environ.get("GOPEX_MTAVG2_FOLD", os.environ.get("GOPEX_MTAVG2_DIAG", "0"))
    return raw.strip().lower() in ("1", "true", "yes")


def ltx_script_prompt_suffix() -> str:
    return LTX_SCRIPT_PROMPT_SUFFIX


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    if mtavg2_diag_enabled():
        out.update(mtavg2_preprocess_extra())
    return out


def mtavg2_preprocess_extra() -> dict[str, Any]:
    return {
        "mtavg2": {
            "enabled": True,
            "benchmark": "MTAVG-Bench 2.0",
            "arxiv": "2605.28035",
            "taxonomy": "acting,atmosphere,cinematography",
            "n_sub_dimensions": 10,
            "n_failure_modes": 45,
            "fold_role": "cinematic_expressiveness_diagnosis",
        }
    }


def attach_mtavg2_to_preprocess_meta(preprocessed_root: str | Path) -> dict[str, Any]:
    root = Path(preprocessed_root).expanduser().resolve()
    meta_path = root / "preprocess_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "missing preprocess_meta.json", "path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(mtavg2_preprocess_extra())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(meta_path)}


def reference_failure_profile(model_label: str = "LTX 2.3") -> dict[str, float]:
    return dict(FIG4_FAILURE_RATE_PCT.get(model_label, FIG4_FAILURE_RATE_PCT["LTX 2.3"]))


def reference_holistic_metrics(model_label: str = "LTX 2.3") -> dict[str, float]:
    return dict(TABLE4_HOLISTIC_METRICS.get(model_label, TABLE4_HOLISTIC_METRICS["LTX 2.3"]))


def diagnose_clip_proxy(
    *,
    script_summary: str = "",
    sub_dim: str = "DP",
    latent_temporal_diff: float | None = None,
    latent_spatial_std: float | None = None,
) -> dict[str, Any]:
    """CPU mock diagnosis: map latent statistics + sub-dim to a plausible failure mode."""
    code = sub_dim.upper()
    info = SUB_DIMENSION_BY_CODE.get(code, SUB_DIMENSION_BY_CODE["DP"])
    modes = [m for m in FAILURE_MODE_BY_ID.values() if m["sub_dim"] == code]
    if not modes:
        modes = list(FAILURE_MODE_BY_ID.values())[:3]

    td = 0.25 if latent_temporal_diff is None else float(latent_temporal_diff)
    ss = 0.5 if latent_spatial_std is None else float(latent_spatial_std)
    severity = min(1.0, max(0.0, 0.55 * td + 0.45 * min(ss, 1.0)))
    idx = min(len(modes) - 1, int(severity * len(modes)))
    chosen = modes[idx]
    ref_fr = reference_failure_profile()["average"]

    return {
        "sub_dimension": code,
        "sub_dimension_name": info["name"],
        "category": info["category"],
        "predicted_failure_mode": chosen["id"],
        "predicted_failure_label": chosen["label"],
        "severity_proxy": round(severity, 4),
        "script_chars": len(script_summary),
        "reference_ltx_failure_rate_avg": ref_fr,
        "diagnosis_confidence_proxy": round(1.0 - abs(severity - ref_fr), 4),
    }


def score_generation_vs_bench(
    *,
    audio_aesthetic: float,
    lip_sync: float,
    av_align: float,
    desync: float,
    ta_align: float,
    tv_align: float,
    latent_temporal_diff: float = 0.3,
) -> dict[str, Any]:
    """Compare clip metrics to Table 4 LTX reference + failure-rate prior."""
    ref = reference_holistic_metrics("LTX 2.3")
    hqi = holistic_quality_index(
        audio_aesthetic=audio_aesthetic,
        lip_sync=lip_sync,
        av_align=av_align,
        desync=desync,
        ta_align=ta_align,
        tv_align=tv_align,
    )
    ref_hqi = holistic_quality_index(
        audio_aesthetic=ref["AudioAesthetic"],
        lip_sync=ref["LipSync"],
        av_align=ref["AVAlign"],
        desync=ref["Desync"],
        ta_align=ref["TAAlign"],
        tv_align=ref["TVAlign"],
    )
    diag = diagnose_clip_proxy(sub_dim="CT", latent_temporal_diff=latent_temporal_diff)
    return {
        "holistic_quality_index": round(hqi, 4),
        "ltx_reference_hqi": round(ref_hqi, 4),
        "delta_vs_ltx_table4": round(hqi - ref_hqi, 4),
        "low_level_metrics": {
            "AudioAesthetic": audio_aesthetic,
            "LipSync": lip_sync,
            "AVAlign": av_align,
            "Desync": desync,
            "TAAlign": ta_align,
            "TVAlign": tv_align,
        },
        "reference_table4": ref,
        "cinematic_diagnosis": diag,
        "note": (
            "Strong Table-4 scores do not imply low cinematic failure rates (paper Sec. 5.2); "
            "use mtavg2 QA for scene-level diagnosis."
        ),
    }


def write_diagnosis_report(
    out_path: str | Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = Path(out_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"ok": True, "written": len(rows), "path": str(path)}
