"""OCL ICC: Implicit Cycle Consistency for video object-centric learning (arXiv:2605.30211).

Bidirectional video OCL with reconstruction-manifold cycle consistency (ICC) vs latent ECC.
DINOv2 backbone and full MOVi/YTVIS training are external.
"""

from ltx_trainer.ocl_icc.analysis import (
    latent_distance,
    manifold_scatter_point,
    reconstruction_consensus,
    slot_diversity,
    slot_temporal_variance,
)
from ltx_trainer.ocl_icc.config import OCLICCConfig
from ltx_trainer.ocl_icc.losses import (
    explicit_cycle_loss,
    hungarian_ecc_loss,
    implicit_cycle_loss,
    reconstruction_loss,
    total_objective,
)
from ltx_trainer.ocl_icc.ltx_bridge import OCLICCLTXBridge, ltx_integration_notes
from ltx_trainer.ocl_icc.metrics import (
    table1_object_discovery,
    table2_efficiency,
    table3_recognition,
    table4_ablation_movc,
    table5_collapse_ytvis,
)
from ltx_trainer.ocl_icc.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    training_step_demo,
)
from ltx_trainer.ocl_icc.slots import SlotAttention, TransitionModule
from ltx_trainer.ocl_icc.streams import StreamOutput, VideoOCLStreams

__all__ = [
    "OCLICCConfig",
    "OCLICCLTXBridge",
    "SlotAttention",
    "StreamOutput",
    "TransitionModule",
    "VideoOCLStreams",
    "benchmark_manifest",
    "evaluation_demo",
    "explicit_cycle_loss",
    "framework_card",
    "hungarian_ecc_loss",
    "implicit_cycle_loss",
    "latent_distance",
    "ltx_integration_notes",
    "manifold_scatter_point",
    "paper_limitations",
    "reconstruction_consensus",
    "reconstruction_loss",
    "slot_diversity",
    "slot_temporal_variance",
    "table1_object_discovery",
    "table2_efficiency",
    "table3_recognition",
    "table4_ablation_movc",
    "table5_collapse_ytvis",
    "total_objective",
    "training_step_demo",
]
