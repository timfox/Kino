"""Known limitations of the CPU EvoGS stub."""

LIMITATIONS = [
    "Numpy/torch stub — no official 3DGS/LapisGS CUDA rasterizer or H100 training loop.",
    "Table 1–5 numbers are paper anchors; synthetic demos validate ordering and refinement math only.",
    "Image pyramid and progressive stages are simulated on random splats, not Mip-NeRF360 captures.",
    "Compression uses uniform 8-bit quant + zlib size proxy, not production zstd bitstreams.",
    "Mixed-resolution foveated traversal (Sec. 3.3) is planned in ltx_plan but not rasterized here.",
]
