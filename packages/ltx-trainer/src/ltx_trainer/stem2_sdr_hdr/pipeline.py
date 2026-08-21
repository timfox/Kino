"""Bridge StEM2 analysis into LTX trainer paper stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.stem2_sdr_hdr._bootstrap import ensure_repo_root

ensure_repo_root()

from gopex_datasets.stem2.pipeline import analyze_frame, evaluate_synthetic, full_report
from gopex_datasets.stem2.synthetic import Stem2Triplet


def analyze_synthetic_scene(scene: str = "desert", *, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
    """Run luminance / color / decision-map metrics on a synthetic StEM2 triplet."""
    return evaluate_synthetic(scene, seed=seed, **kwargs)


def cinema_mapping_report() -> dict[str, Any]:
    """Paper tables plus synthetic sanity checks (no StEM2 download)."""
    return full_report()


def isotonic_readiness_from_triplet(scene: str = "desert", *, seed: int = 0) -> float:
    """Map synthetic isotonic R² to a 0–1 cinema-mapping readiness proxy."""
    metrics = analyze_synthetic_scene(scene, seed=seed, height=48, width=64)
    return float(min(1.0, max(0.0, (metrics["r2"] - 0.85) / 0.15)))


def analyze_custom_triplet(exr: Any, sdr: Any, hdr: Any, *, scene: str = "custom") -> dict[str, Any]:
    """Analyze caller-supplied EXR/SDR/HDR tensors."""
    return analyze_frame(Stem2Triplet(exr=exr, sdr=sdr, hdr=hdr, scene=scene))
