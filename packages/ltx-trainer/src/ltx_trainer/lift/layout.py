"""LiFT pipeline layout and limitations."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Performance is coupled to the underlying 2D slice generator or translator quality.",
    "LiFT-U provides distributional guarantees, not instance-specific volume fidelity.",
    "LiFT-C assumes paired, accurately registered source–target volumes.",
    "Evaluation is limited to brain MR/CT benchmarks; clinical deployability is not claimed.",
    "Missing-MR comparison to cWDM uses published aggregates without paired statistical testing.",
)

PIPELINE_STAGES: tuple[str, ...] = (
    "Per-slice 2D synthesis (frozen G2D for LiFT-U, or encoder–decoder for LiFT-C)",
    "Depth-indexed trajectory: Fourier γ(d) + mapper Mϕ (LiFT-U) or BiGRU z-mixer (LiFT-C)",
    "Stack slices → volume V̂",
    "Volume supervision: tri-planar drifting (LiFT-U) or Lpixel + Lsimilarity + Lspatial (LiFT-C)",
)

LIFT_U_STAGES: tuple[str, ...] = (
    "Pretrain & freeze 2D axial slice generator G2D",
    "Train depth mapper Mϕ(z, γ(d)) → per-slice conditioning cd",
    "Tri-planar drift: align Efeat(π(V̂)) with real banks on axial/coronal/sagittal",
)

LIFT_C_STAGES: tuple[str, ...] = (
    "Encode source slices → bottleneck hd, pool to bd",
    "Bidirectional GRU over b1:D with depth encoding → context cd",
    "Decode ˆyd = Dθ(hd, cd); two-pass inference at native resolution (Alg. 2)",
)
