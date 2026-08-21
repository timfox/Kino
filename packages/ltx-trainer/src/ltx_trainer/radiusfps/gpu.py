"""RadiusFPS-G GPU fusion kernel summary (§4)."""

from __future__ import annotations

from typing import Any


def fusion_kernel_card() -> dict[str, Any]:
    return {
        "voxelization": "parallel map → radix sort → AoS-to-SoA gather → active-voxel compaction (Alg. 2)",
        "fusion_kernel_1": "two-level warp argmax: Distv → best voxel → Distp segment → sample point (Alg. 3)",
        "fusion_kernel_2": "one-block-one-voxel: radius prune or coalesced point update + block max (Alg. 4)",
        "memory": "~half QuickFPS GPU footprint; coalesced SoA access",
        "host_device": "full pipeline resident on GPU; minimal sync per iteration",
    }


def gpu_pipeline_stages() -> list[dict[str, str]]:
    return [
        {"stage": "Phase 1", "name": "Parallel voxel mapping + radix sort"},
        {"stage": "Phase 2", "name": "Gather AoS→SoA"},
        {"stage": "Phase 3", "name": "Active voxel offsets/counts"},
        {"stage": "Phase 4", "name": "Global distance init + block reduce Distv"},
        {"stage": "Loop", "name": "Fusion K1 (select) + Fusion K2 (filter/update)"},
    ]
