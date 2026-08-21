"""Paper tables and benchmark metrics (Tabs. 1–5, Figs. 1–4)."""

from __future__ import annotations

from typing import Any


def table_probe_accuracy_by_layer() -> list[dict[str, Any]]:
    """Layer-wise probe accuracy (5-fold CV); Fig. 1 / Tab. 1 (ε=0.05 PEZ threshold)."""
    from ltx_trainer.physics_steering.experiments import full_layer_accuracy_reference
    from ltx_trainer.physics_steering.pez import identify_pez_layers

    acc = full_layer_accuracy_reference()
    pez = set(identify_pez_layers(acc, epsilon=0.05))
    std_by_layer = {5: 0.013, 0: 0.035, 1: 0.045, 2: 0.053, 3: 0.027, 7: 0.027, 11: 0.025}
    return [
        {
            "layer": layer,
            "val_acc": val,
            "std": std_by_layer.get(layer, 0.03),
            "in_pez": layer in pez,
        }
        for layer, val in sorted(acc.items())
    ]


def layer_accuracy_dict() -> dict[int, float]:
    from ltx_trainer.physics_steering.experiments import full_layer_accuracy_reference

    return full_layer_accuracy_reference()


def table_alpha_sweep() -> list[dict[str, float | int]]:
    """Table 2 — steering strength at l*=5."""
    return [
        {"alpha": -20, "flip_rate": 0.75, "p_impossible": 0.0, "cosine_shift": -0.743},
        {"alpha": -10, "flip_rate": 0.75, "p_impossible": 0.0, "cosine_shift": -0.754},
        {"alpha": -5, "flip_rate": 0.75, "p_impossible": 0.0, "cosine_shift": -0.766},
        {"alpha": 0, "flip_rate": 0.0, "p_impossible": 0.595, "cosine_shift": 0.0},
        {"alpha": 5, "flip_rate": 0.25, "p_impossible": 1.0, "cosine_shift": 0.775},
        {"alpha": 10, "flip_rate": 0.25, "p_impossible": 1.0, "cosine_shift": 0.773},
        {"alpha": 20, "flip_rate": 0.25, "p_impossible": 1.0, "cosine_shift": 0.775},
    ]


def table_layer_ablation() -> list[dict[str, float | int | bool]]:
    """Table 3 — inject PEZ CAV at each layer (α=10)."""
    return [
        {"injection_layer": 0, "in_pez": True, "flip_rate": 0.25, "directional_purity": 0.376},
        {"injection_layer": 1, "in_pez": True, "flip_rate": 0.25, "directional_purity": 0.381},
        {"injection_layer": 2, "in_pez": False, "flip_rate": 0.25, "directional_purity": 0.432},
        {"injection_layer": 3, "in_pez": False, "flip_rate": 0.25, "directional_purity": 0.471},
        {"injection_layer": 4, "in_pez": False, "flip_rate": 0.25, "directional_purity": 0.719},
        {"injection_layer": 5, "in_pez": True, "flip_rate": 0.25, "directional_purity": 1.0},
        {"injection_layer": 6, "in_pez": False, "flip_rate": 0.0, "directional_purity": 0.0},
        {"injection_layer": 7, "in_pez": False, "flip_rate": 0.0, "directional_purity": 0.0},
        {"injection_layer": 8, "in_pez": False, "flip_rate": 0.0, "directional_purity": 0.0},
        {"injection_layer": 9, "in_pez": False, "flip_rate": 0.0, "directional_purity": 0.0},
        {"injection_layer": 10, "in_pez": False, "flip_rate": 0.0, "directional_purity": 0.0},
        {"injection_layer": 11, "in_pez": False, "flip_rate": 0.0, "directional_purity": 0.0},
    ]


def table_block_cav_disentanglement() -> dict[str, Any]:
    """Table 4 — per-principle probes and CAV angles at l*=5."""
    return {
        "probe_accuracy": {
            "O1": 0.742,
            "O2": 0.700,
            "O3": 0.758,
        },
        "cav_angles_deg": {
            "O1_vs_O2": 75.7,
            "O1_vs_O3": 76.3,
            "O2_vs_O3": 86.1,
        },
    }


def table_subspace_orthogonality() -> list[dict[str, float | str]]:
    """Table 5 — physics vs motion vs random."""
    return [
        {"concept_pair": "Physics vs motion direction", "angle_deg": 90.0},
        {"concept_pair": "Physics vs random unit vector", "angle_deg": 85.0},
        {"concept_pair": "Mean physics orthogonality", "angle_deg": 87.5},
    ]
