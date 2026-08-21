"""Clinical layout and study limitations."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Pilot cohort: 30 patients (9 readmitted, 21 not) — feasibility evidence, not definitive validation.",
    "Class imbalance limits generalization; nested CV mitigates but does not eliminate variance.",
    "Requires standardized six-view POCUS protocol and multi-day inpatient imaging.",
    "Follow IRB and de-identification requirements for clinical ultrasound data transfer.",
)

VIEW_ANATOMY: dict[str, str] = {
    "Left-1": "Upper anterior (L1)",
    "Left-2": "Lateral (L2)",
    "Left-3": "Dependent posterior (L3) — strongest prognostic signal",
    "Right-1": "Upper anterior (R1)",
    "Right-2": "Lateral (R2)",
    "Right-3": "Dependent posterior (R3) — strongest prognostic signal",
}

PIPELINE_STAGES: tuple[str, ...] = (
    "Acquire six standardized LUS views (R/L 1–3) at ≥2 hospitalization time points",
    "Encode B-mode clips with frozen TSM–ResNet-18 → 512-D embeddings",
    "Form temporal difference features Δ = Day2 − Day1 per view",
    "Concatenate six-view Δ embeddings → patient-level vector",
    "Classify 30-day readmission with nested CV downstream model (MLP best)",
)
