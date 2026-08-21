"""Per-project training sample weights for merged_native-style latent trees."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

ProjectSamplingMode = Literal["uniform", "balanced", "sqrt"]

# First path component under ``latents/<project>/…`` for merged precompute trees.
_MERGE_META_SKIP = frozenset({"ltx_manifest", "conditions", "audio_latents", "latents"})


def project_slug_from_latent_path(rel_path: Path | str) -> str:
    """Infer catalog project slug from a latent shard relative path."""
    parts = Path(rel_path).parts
    if len(parts) >= 2 and parts[0] not in _MERGE_META_SKIP:
        return parts[0]
    if parts:
        stem = parts[0]
        if stem not in _MERGE_META_SKIP:
            return stem
    return "_root"


def build_project_groups(sample_rel_paths: list[Path]) -> dict[str, list[int]]:
    """Map project slug → dataset indices."""
    groups: dict[str, list[int]] = {}
    for idx, rel in enumerate(sample_rel_paths):
        slug = project_slug_from_latent_path(rel)
        groups.setdefault(slug, []).append(idx)
    return groups


def compute_project_sample_weights(
    groups: dict[str, list[int]],
    *,
    mode: ProjectSamplingMode = "uniform",
    num_samples: int | None = None,
) -> list[float]:
    """Return per-index weights aligned with ``0 .. num_samples-1``."""
    if not groups:
        return []
    n = num_samples if num_samples is not None else max(max(idxs) for idxs in groups.values()) + 1
    weights = [0.0] * n

    if mode == "uniform":
        total = sum(len(idxs) for idxs in groups.values())
        if total <= 0:
            return weights
        w = 1.0 / float(total)
        for idxs in groups.values():
            for i in idxs:
                weights[i] = w
        return weights

    if mode == "balanced":
        n_proj = len(groups)
        for idxs in groups.values():
            if not idxs:
                continue
            w = 1.0 / float(n_proj * len(idxs))
            for i in idxs:
                weights[i] = w
        return weights

    if mode == "sqrt":
        per_project = {slug: 1.0 / math.sqrt(float(len(idxs))) for slug, idxs in groups.items() if idxs}
        denom = sum(per_project[slug] * len(groups[slug]) for slug in per_project)
        if denom <= 0:
            return weights
        for slug, idxs in groups.items():
            w = per_project.get(slug, 0.0) / denom
            for i in idxs:
                weights[i] = w
        return weights

    raise ValueError(f"Unknown project sampling mode: {mode!r}")


def summarize_project_groups(groups: dict[str, list[int]]) -> list[dict[str, float | int | str]]:
    """Human-readable project counts and uniform vs balanced share."""
    total = sum(len(v) for v in groups.values())
    n_proj = len(groups)
    rows: list[dict[str, float | int | str]] = []
    for slug in sorted(groups, key=lambda s: (-len(groups[s]), s)):
        count = len(groups[slug])
        rows.append(
            {
                "project": slug,
                "clips": count,
                "uniform_pct": round(100.0 * count / total, 2) if total else 0.0,
                "balanced_pct": round(100.0 / n_proj, 2) if n_proj else 0.0,
            }
        )
    return rows
