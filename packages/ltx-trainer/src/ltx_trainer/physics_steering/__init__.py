"""Physics steering via CAVs at the VideoMAE PEZ (Alam, arXiv:2605.24322).

Reference: mean-pool probing (Eq. 1), PEZ selection (Eq. 3), CAV normalization (Eq. 4),
inference steering (Eq. 5), and IntPhys evaluation metrics. **VideoMAE weights**, forward
hooks, and **sklearn** probes are **not** bundled — plug into your activation pipeline.
"""

from ltx_trainer.physics_steering.config import PhysicsSteeringConfig
from ltx_trainer.physics_steering.intphys import INTPHYS_BLOCK_DESCRIPTIONS, synthetic_intphys_features
from ltx_trainer.physics_steering.ltx_bridge import PhysicsSteeringLTXBridge, ltx_integration_notes
from ltx_trainer.physics_steering.ltx_validation import audit_npz_primary_layer, audit_pooled_hidden, bridge_from_bundle
from ltx_trainer.physics_steering.videomae_steering_check import check_video_steering
from ltx_trainer.physics_steering.metrics import (
    directional_purity,
    flip_rate,
    representation_drift,
    score_delta,
)
from ltx_trainer.physics_steering.paper_tables import (
    layer_accuracy_dict,
    table_alpha_sweep,
    table_block_cav_disentanglement,
    table_layer_ablation,
    table_probe_accuracy_by_layer,
    table_subspace_orthogonality,
)
from ltx_trainer.physics_steering.pez import identify_pez_layers, top_pez_layers
from ltx_trainer.physics_steering.activations_io import load_layer_activations, save_layer_activations
from ltx_trainer.physics_steering.bundle import export_steering_bundle, load_steering_bundle
from ltx_trainer.physics_steering.collect import collect_intphys_activations
from ltx_trainer.physics_steering.paper_report import compare_experiment_to_paper, format_report_markdown
from ltx_trainer.physics_steering.inference_hooks import PhysicsSteeringHookManager
from ltx_trainer.physics_steering.experiments import (
    fit_block_cavs,
    fit_pez_cavs,
    fit_physics_cav,
    full_layer_accuracy_reference,
    kfold_probe_accuracy,
    run_alpha_sweep,
    run_full_experiment,
    run_layer_ablation,
    run_synthetic_paper_benchmark,
    stratified_split_indices,
    subspace_orthogonality_report,
    synthetic_multilayer_features,
)
from ltx_trainer.physics_steering.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    training_step_demo,
)
from ltx_trainer.physics_steering.probe import (
    cav_from_weights,
    fit_logistic_probe,
    fit_probe_with_pca,
    logistic_predict_proba,
)
from ltx_trainer.physics_steering.representation import batch_mean_pool, mean_pool_hidden
from ltx_trainer.physics_steering.steering import (
    angle_between,
    iterative_orthogonal_probe_accuracies,
    project_out_direction,
    steer_batch_scores,
    steer_hidden_states,
    steer_representation,
)

__all__ = [
    "INTPHYS_BLOCK_DESCRIPTIONS",
    "PhysicsSteeringConfig",
    "PhysicsSteeringHookManager",
    "PhysicsSteeringLTXBridge",
    "angle_between",
    "audit_npz_primary_layer",
    "audit_pooled_hidden",
    "bridge_from_bundle",
    "check_video_steering",
    "compare_experiment_to_paper",
    "batch_mean_pool",
    "benchmark_manifest",
    "collect_intphys_activations",
    "cav_from_weights",
    "directional_purity",
    "evaluation_demo",
    "export_steering_bundle",
    "PhysicsSteeringHookManager",
    "fit_block_cavs",
    "fit_pez_cavs",
    "fit_logistic_probe",
    "fit_physics_cav",
    "fit_probe_with_pca",
    "full_layer_accuracy_reference",
    "flip_rate",
    "format_report_markdown",
    "framework_card",
    "identify_pez_layers",
    "load_steering_bundle",
    "kfold_probe_accuracy",
    "load_layer_activations",
    "iterative_orthogonal_probe_accuracies",
    "layer_accuracy_dict",
    "logistic_predict_proba",
    "ltx_integration_notes",
    "mean_pool_hidden",
    "paper_limitations",
    "project_out_direction",
    "representation_drift",
    "run_alpha_sweep",
    "run_full_experiment",
    "run_layer_ablation",
    "run_synthetic_paper_benchmark",
    "save_layer_activations",
    "score_delta",
    "stratified_split_indices",
    "subspace_orthogonality_report",
    "synthetic_multilayer_features",
    "steer_batch_scores",
    "steer_hidden_states",
    "steer_representation",
    "synthetic_intphys_features",
    "table_alpha_sweep",
    "table_block_cav_disentanglement",
    "table_layer_ablation",
    "table_probe_accuracy_by_layer",
    "table_subspace_orthogonality",
    "top_pez_layers",
    "training_step_demo",
]
