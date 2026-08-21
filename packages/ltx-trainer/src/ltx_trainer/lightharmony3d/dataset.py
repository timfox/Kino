"""LH3D-Bench and training corpus metadata (Sec. 4–5)."""

from __future__ import annotations

LH3D_BENCH = {
    "lh3d_ku": {"scenes": 4, "test_pairs": 20, "source": "Kujiale indoor + PolyHaven objects"},
    "lh3d_blender": {"scenes": 7, "objects_per_scene": 9, "test_images": 315},
}

TRAINING_SOURCES = {
    "hidrovqa": 411,
    "chug": 428,
    "live_tmhdr": 40,
}

BASELINES = ("GIGS", "GaussianEditor", "MV-CoLight", "GaSLight")


def dataset_summary() -> dict[str, object]:
    return {
        "benchmark": LH3D_BENCH,
        "training_sources": TRAINING_SOURCES,
        "reconstruction": "MILo hybrid 3DGS–mesh",
        "gen_env": "Flux.1 Kontext + DreamBooth LoRA (800+ HDRIs)",
        "eval_real": "Mip-NeRF360",
        "baselines": list(BASELINES),
    }
