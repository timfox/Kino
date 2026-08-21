"""SpecX tiers, tasks, and limitations."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Primarily simulated spectra — sim-to-real gap; Exp subset is UV+MS only (432 molecules).",
    "Chemical space biased toward synthetic/drug-like compounds; limited natural products.",
    "Fluorescence included in dataset but deferred from Large/Small benchmark splits.",
    "MLLM evaluation is zero-shot without formula priors or constrained SMILES decoding.",
    "Baselines use vanilla Transformer; SOTA per-modality models not yet on leaderboard.",
)

TIER_DESCRIPTIONS: dict[str, str] = {
    "Large": "~1M molecules for pretraining and Tasks (1)–(3); seven modalities (no FL in splits).",
    "Small": "4,496 strictly aligned multispectral molecules for Task (4) QA.",
    "Exp": "432 molecules with experimental UV-Vis and MS for real-world generalization.",
}

TASK_DESCRIPTIONS: dict[str, str] = {
    "spectra_to_smiles": "Structure elucidation: spectral text → SMILES (Top-1/5/10, random + scaffold).",
    "functional_group_prediction": "37 SMARTS functional groups; XGBoost vs 1D-CNN; macro-F1.",
    "smiles_to_spectra": "Inverse simulation: SMILES → spectrum tokens; cosine + token accuracy.",
    "qa_smiles_inference": "MLLM zero-shot SMILES from spectra (Small/Exp subsets).",
    "qa_functional_groups": "MLLM functional group QA vs specialized baselines.",
}

BENCHMARK_COMPARISON: list[dict[str, str]] = [
    {"name": "NovoBench", "scale": "small", "multi_spec": "✗", "experimental": "✗"},
    {"name": "MolPuzzle", "scale": "small", "multi_spec": "✗", "experimental": "✗"},
    {"name": "Multimodal Spec", "scale": "medium", "multi_spec": "partial", "experimental": "partial"},
    {"name": "MassSpecGym", "scale": "medium", "multi_spec": "✗", "experimental": "✓"},
    {"name": "SpecX", "scale": "1.7M", "multi_spec": "aligned", "experimental": "✓"},
]
