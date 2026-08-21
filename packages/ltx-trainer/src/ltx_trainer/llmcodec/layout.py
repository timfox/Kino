"""LLMCodec stub limitations."""

LIMITATIONS = (
    "No VVenC/VVdeC binary integration, no real LLaMA/Qwen weight tensors, "
    "no calibration-data affine learning loop, and no lm-eval harness runs. "
    "Codec compression is a QP-scaled numpy proxy (PSNR/bitrate), not ITU-T bitstreams."
)
