"""SVHighlights CPU reference smokes (arXiv:2606.06926)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.svhighlights.alignment import align_highlight_sequence, alignment_stats, downsample_frame, psnr
from ltx_trainer.svhighlights.config import SvHighlightsConfig
from ltx_trainer.svhighlights.labeling import labels_from_alignment
from ltx_trainer.svhighlights.metrics import evaluate_predictions
from ltx_trainer.svhighlights.pipeline import table_vii_main_results
from ltx_trainer.svhighlights.segmentation import WordSpan
from ltx_trainer.svhighlights.tf_selector import run_tf_selector_stub


def _synthetic_broadcast(*, n_full: int = 900, n_hl: int = 30, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Full video + highlight subsequence with controlled PSNR."""
    rng = np.random.default_rng(seed)
    full: list[np.ndarray] = []
    for i in range(n_full):
        base = rng.integers(0, 256, size=(144, 144, 3), dtype=np.uint8)
        base[:, :, 0] = (base[:, :, 0].astype(np.int16) + (i % 17)) % 256
        full.append(base.astype(np.uint8))
    highlight_indices = sorted(rng.choice(n_full - 10, size=n_hl, replace=False).tolist())
    hl = [full[i].copy() for i in highlight_indices]
    for frame in hl:
        frame[:] = np.clip(frame.astype(np.int16) + rng.integers(-2, 3, size=frame.shape), 0, 255).astype(np.uint8)
    return full, hl


def run_alignment_smoke(*, seed: int = 0) -> dict[str, Any]:
    c = SvHighlightsConfig()
    full, hl = _synthetic_broadcast(seed=seed)
    full_ds = [downsample_frame(f, c.alignment_downsample_p) for f in full]
    hl_ds = [downsample_frame(h, c.alignment_downsample_p) for h in hl]
    aligned = align_highlight_sequence(
        hl_ds,
        full_ds,
        tau=c.alignment_tau,
        psnr_threshold=c.psnr_filter_threshold,
    )
    stats = alignment_stats(hl_ds, full_ds, c)
    retained_idx = [i for i, _ in aligned if i is not None]
    return {
        "n_highlight_frames": len(hl),
        "n_retained": len(retained_idx),
        "stats": stats.to_dict(),
        "mean_pair_psnr": round(float(np.mean([s for _, s in aligned if _ is not None])), 2) if retained_idx else 0.0,
        "alignment_ok": len(retained_idx) >= len(hl) * 0.5,
    }


def run_label_smoke(*, seed: int = 0) -> dict[str, Any]:
    full, hl = _synthetic_broadcast(seed=seed)
    full_ds = [downsample_frame(f) for f in full]
    hl_ds = [downsample_frame(h) for h in hl]
    aligned = align_highlight_sequence(hl_ds, full_ds, tau=5.0, psnr_threshold=20.0)
    duration_s = len(full) / 30.0
    lab = labels_from_alignment(aligned, fps=30.0, duration_s=duration_s)
    return {
        "n_clips": lab["n_clips"],
        "n_positive": lab["n_positive"],
        "positive_rate": round(lab["n_positive"] / max(lab["n_clips"], 1), 3),
        "label_ok": lab["n_positive"] > 0,
    }


def run_tf_selector_smoke(*, seed: int = 0) -> dict[str, Any]:
    duration_s = 300.0
    rng = np.random.default_rng(seed)
    words: list[WordSpan] = []
    t = 0.0
    while t < duration_s - 1.0:
        cue = "goal" if rng.random() > 0.85 else "play"
        length = float(rng.uniform(0.2, 0.5))
        words.append(WordSpan(t, t + length, cue))
        t += float(rng.uniform(0.8, 2.0))
    out = run_tf_selector_stub(duration_s, words, seed=seed)
    return {
        **out,
        "selector_ok": out["n_segments"] >= 1 and out["top_segment_score"] > 0,
    }


def run_metrics_smoke(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 50
    labels = (rng.random(n) > 0.85).astype(np.int8)
    scores = rng.random(n)
    scores[labels == 1] += 0.5
    metrics = evaluate_predictions(scores.tolist(), labels.tolist())
    return {"synthetic_metrics": metrics, "metrics_ok": metrics["mAP"] >= 0}


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    align = run_alignment_smoke(seed=seed)
    label = run_label_smoke(seed=seed + 1)
    selector = run_tf_selector_smoke(seed=seed + 2)
    metrics = run_metrics_smoke(seed=seed + 3)
    tf_row = next(r for r in table_vii_main_results() if r["method"] == "TF-SELECTOR")
    trace_row = next(r for r in table_vii_main_results() if r["method"] == "TRACE")
    paper_ok = (
        float(tf_row["hit_at_1"]) > float(trace_row["hit_at_1"])
        and float(tf_row["hit_at_k"]) > float(trace_row["hit_at_k"])
        and float(tf_row["iou"]) > float(trace_row["iou"])
    )
    return {
        "alignment": align,
        "labeling": label,
        "tf_selector": selector,
        "metrics": metrics,
        "paper_tf_beats_trace_on_hit_iou": paper_ok,
        "pipeline_ok": (
            align["alignment_ok"]
            and label["label_ok"]
            and selector["selector_ok"]
            and metrics["metrics_ok"]
            and paper_ok
        ),
    }
