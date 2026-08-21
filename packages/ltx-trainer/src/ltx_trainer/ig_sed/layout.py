"""Scope limitations for IG-SED stub."""

LIMITATIONS = (
    "Reference stub only: no PANNs CNN14 weights, Captum, DESED/Scaper data, or PyTorch training.",
    "Integrated gradients use a toy 1D Riemann sum over a synthetic score function, not CNN14.",
    "Benchmark numbers are fixed paper excerpts (Tables I–III, threshold sensitivity), not reproduced runs.",
)
