"""LuckyHDR limitations (trainable reimplementation, not official weights)."""

LIMITATIONS = [
    "Trainable PyTorch reimplementation (~66K params target); not Princeton/Adobe official weights.",
    "Synthetic SI-HDR training smoke only; real smartphone bursts need domain calibration.",
    "Iterative merge assumes radiometrically aligned brackets after shift stage.",
]
