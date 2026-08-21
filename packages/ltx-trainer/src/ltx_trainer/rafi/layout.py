"""Scope notes for RaFI reference stub."""

LIMITATIONS = (
    "Reference stub only: no CUDA-aware MPI, no CUB device sort, no OptiX/OWL. "
    "Device emit and host forwardRays() are CPU/torch smokes that mirror the paper "
    "algorithm (sort-by-dest, tally, Alltoall/Alltoallv). Paper bandwidth numbers are excerpts."
)
