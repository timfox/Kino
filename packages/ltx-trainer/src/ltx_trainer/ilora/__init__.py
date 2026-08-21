"""iLoRA — Bayesian graph-conditioned LoRA (arXiv:2605.30179)."""

from ltx_trainer.ilora.config import ILORAConfig
from ltx_trainer.ilora.graph import (
    PoissonLaplaceGraphBranch,
    edge_features,
    laplace_kl,
    matched_poisson_rate,
    poisson_kl,
)
from ltx_trainer.ilora.gnn import GraphEncoder, normalize_adjacency
from ltx_trainer.ilora.loss import ilora_total_loss, prediction_loss
from ltx_trainer.ilora.lora import GraphHyperLoRA, StaticLoRA
from ltx_trainer.ilora.ltx_bridge import ILORALTXAdapterBridge, ltx_peft_integration_notes
from ltx_trainer.ilora.metrics import (
    table1_molweni,
    table2_graph_error,
    table3_ibd_diagnosis,
    table4_ablation,
    table5_tabular_baselines,
    table8_inference_cost,
)
from ltx_trainer.ilora.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    training_step_demo,
)

__all__ = [
    "ILORAConfig",
    "ILORALTXAdapterBridge",
    "GraphEncoder",
    "GraphHyperLoRA",
    "PoissonLaplaceGraphBranch",
    "StaticLoRA",
    "benchmark_manifest",
    "edge_features",
    "evaluation_demo",
    "framework_card",
    "ilora_total_loss",
    "laplace_kl",
    "ltx_peft_integration_notes",
    "matched_poisson_rate",
    "normalize_adjacency",
    "paper_limitations",
    "poisson_kl",
    "prediction_loss",
    "table1_molweni",
    "table2_graph_error",
    "table3_ibd_diagnosis",
    "table4_ablation",
    "table5_tabular_baselines",
    "table8_inference_cost",
    "training_step_demo",
]
