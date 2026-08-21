"""SwanSphere scope and limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No pretrained SwanSphere Spatial LM, LocDiT, or FOA-VAE weights in this stub.",
    "No VideoMAE-V2 / AudioMAE / Gemini 2.5 Pro runtime in-tree.",
    "Streaming latency numbers are paper-reported anchors, not measured on local hardware.",
    "Sphere360 / panoramic ERP video hooks to LTX are planning-only.",
)
