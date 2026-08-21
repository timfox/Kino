"""Scope and limitations (paper-aligned)."""

LIMITATIONS: tuple[str, ...] = (
    "Pointwise correlation often below operational HRRR on most variables and leads.",
    "LCM distillation trades some DDPM fidelity for 4–25 step inference.",
    "Single deterministic realisation reported; ensemble CRPS not yet in this reference build.",
    "Zero-shot skill drops on climatologically distant domains (e.g. Germany vs CONUS).",
    "Training used calendar year 2021 only in the initial open-weights release.",
)
