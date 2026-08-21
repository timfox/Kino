"""MARI — Multi-Adapter Representation Interventions via Energy Calibration (arXiv:2605.28722)."""

from ltx_trainer.mari.adapters import MultiAdapterBank, competitive_train_step
from ltx_trainer.mari.baselines import (
    INFERENCE_OVERHEAD_NOTE,
    TABLE1,
    TABLE2_ABLATION,
    TABLE3_TRANSFER,
    TABLE4_LATENCY,
)
from ltx_trainer.mari.config import MARIConfig
from ltx_trainer.mari.energy import (
    EnergyGate,
    calibrate_energy_threshold,
    propagation_energy,
    probe_delta,
    simulate_post_injection_layers,
)
from ltx_trainer.mari.inference import MARIInferenceResult, MARIPipeline
from ltx_trainer.mari.low_rank import LowRankAdapter, init_adapter
from ltx_trainer.mari.losses import mc_loss_from_scores
from ltx_trainer.mari.mock import evaluation_smoke
from ltx_trainer.mari.pca import PCASubspace, fit_pca
from ltx_trainer.mari.pipeline import (
    MCExample,
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
    synthetic_corpus,
)
from ltx_trainer.mari.routing import inference_entropy_route, training_winner, usage_balance_penalty
from ltx_trainer.mari.theory import improvement_condition, non_applicable_energy_bound, routing_risk_bound
from ltx_trainer.mari.training import CompetitiveTrainReport, demo_train as demo_train_competitive, train_competitive_epoch

__all__ = [
    "INFERENCE_OVERHEAD_NOTE",
    "MARIConfig",
    "MARIInferenceResult",
    "MARIPipeline",
    "MCExample",
    "MultiAdapterBank",
    "TABLE1",
    "TABLE2_ABLATION",
    "TABLE3_TRANSFER",
    "TABLE4_LATENCY",
    "EnergyGate",
    "LowRankAdapter",
    "PCASubspace",
    "benchmarks_bundle",
    "calibrate_energy_threshold",
    "CompetitiveTrainReport",
    "competitive_train_step",
    "demo_train_competitive",
    "train_competitive_epoch",
    "evaluation_demo",
    "evaluation_smoke",
    "fit_pca",
    "framework_card",
    "improvement_condition",
    "inference_entropy_route",
    "init_adapter",
    "knowledge_card",
    "mc_loss_from_scores",
    "non_applicable_energy_bound",
    "probe_delta",
    "propagation_energy",
    "routing_risk_bound",
    "simulate_post_injection_layers",
    "synthetic_corpus",
    "training_winner",
    "usage_balance_penalty",
]
