"""ST-SFLora semantic token split federated LoRA stub (arXiv:2605.26120)."""

from ltx_trainer.st_sflora.config import StSfloraConfig
from ltx_trainer.st_sflora.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.st_sflora.mock import evaluation_smoke
from ltx_trainer.st_sflora.optimize import ClientAllocation, ClientLinkState, alternating_optimize
from ltx_trainer.st_sflora.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.st_sflora.radio import (
    client_selected,
    standing_time_s,
    uplink_energy_j,
    uplink_latency_s,
    uplink_rate_bps,
)
from ltx_trainer.st_sflora.ste import semantic_transmission_efficiency
from ltx_trainer.st_sflora.tables import headline_results, table1_top1_accuracy, table2_client_overhead
from ltx_trainer.st_sflora.tokens import (
    batch_token_importance,
    cls_to_patch_attention,
    cumulative_semantic_retention,
    merge_discarded_tokens,
    payload_bits,
)

__all__ = [
    "LIMITATIONS",
    "PIPELINE_STAGES",
    "ClientAllocation",
    "ClientLinkState",
    "StSfloraConfig",
    "alternating_optimize",
    "batch_token_importance",
    "benchmarks_bundle",
    "client_selected",
    "cls_to_patch_attention",
    "cumulative_semantic_retention",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "merge_discarded_tokens",
    "payload_bits",
    "semantic_transmission_efficiency",
    "standing_time_s",
    "table1_top1_accuracy",
    "table2_client_overhead",
    "uplink_energy_j",
    "uplink_latency_s",
    "uplink_rate_bps",
]
