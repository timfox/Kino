"""SwarmHarness: decentralised HarnessAPI skill swarms (Jose, arXiv:2605.28764)."""

from ltx_trainer.swarmharness.bootstrap import (
    BootstrapConfig,
    bootstrap_card,
    parse_peer_list,
)
from ltx_trainer.swarmharness.config import (
    ResourceVector,
    RouterWeights,
    SwarmHarnessConfig,
    SwarmNode,
    normalise_weights,
)
from ltx_trainer.swarmharness.credit import (
    AttributionResult,
    GenesisState,
    allocate_credits,
    attribution_summary,
    grant_node_genesis,
    grant_submitter_genesis_task,
    leave_one_out_quality_proxy,
    shapley_monte_carlo,
    shapley_standard_error,
    single_node_attribution,
    swarm_credit_attribution,
    trust_decay,
    unlock_genesis_after_serve,
    update_trust_scores,
)
from ltx_trainer.swarmharness.harness_bridge import (
    SkillManifest,
    discover_skills,
    harness_bridge_card,
    node_from_skill_folder,
)
from ltx_trainer.swarmharness.identity import (
    CreditReceipt,
    NodeIdentity,
    mine_registration_pow,
    sign_credit_receipt,
    verify_credit_receipt,
    verify_registration_pow,
)
from ltx_trainer.swarmharness.ledger import LedgerEntry, LocalCreditLedger
from ltx_trainer.swarmharness.agent import SwarmAgent, demo_autonomous_agent
from ltx_trainer.swarmharness.kademlia import closest_nodes, xor_distance
from ltx_trainer.swarmharness.marketplace import SkillListing, SkillMarketplace
from ltx_trainer.swarmharness.orchestrator import LocalSwarm, demo_local_swarm
from ltx_trainer.swarmharness.pipeline import (
    benchmarks_bundle,
    bootstrap_mechanisms,
    decentralisation_phases,
    demo_swarm_nodes,
    evaluation_demo,
    framework_card,
    knowledge_card,
    table_pheromone_feedback,
)
from ltx_trainer.swarmharness.tables import (
    deployment_phases,
    open_challenges,
    positioning_table,
    security_surface,
)
from ltx_trainer.swarmharness.registry import (
    CapabilityAdvertisement,
    SwarmRegistry,
    skill_dht_key,
)
from ltx_trainer.swarmharness.router import (
    SwarmTask,
    rank_candidates,
    route_task,
    route_top_k,
    routing_report,
    utility_score,
)

__all__ = [
    "AttributionResult",
    "BootstrapConfig",
    "CapabilityAdvertisement",
    "CreditReceipt",
    "GenesisState",
    "LedgerEntry",
    "LocalCreditLedger",
    "LocalSwarm",
    "NodeIdentity",
    "SkillManifest",
    "ResourceVector",
    "RouterWeights",
    "SwarmHarnessConfig",
    "SwarmNode",
    "SwarmRegistry",
    "SwarmTask",
    "allocate_credits",
    "attribution_summary",
    "SkillListing",
    "SkillMarketplace",
    "SwarmAgent",
    "bootstrap_card",
    "benchmarks_bundle",
    "bootstrap_mechanisms",
    "closest_nodes",
    "demo_autonomous_agent",
    "demo_local_swarm",
    "discover_skills",
    "decentralisation_phases",
    "demo_swarm_nodes",
    "evaluation_demo",
    "deployment_phases",
    "framework_card",
    "grant_node_genesis",
    "knowledge_card",
    "open_challenges",
    "positioning_table",
    "security_surface",
    "harness_bridge_card",
    "mine_registration_pow",
    "node_from_skill_folder",
    "parse_peer_list",
    "grant_submitter_genesis_task",
    "leave_one_out_quality_proxy",
    "normalise_weights",
    "rank_candidates",
    "route_task",
    "route_top_k",
    "routing_report",
    "shapley_monte_carlo",
    "shapley_standard_error",
    "single_node_attribution",
    "sign_credit_receipt",
    "skill_dht_key",
    "swarm_credit_attribution",
    "verify_credit_receipt",
    "verify_registration_pow",
    "table_pheromone_feedback",
    "trust_decay",
    "unlock_genesis_after_serve",
    "update_trust_scores",
    "utility_score",
    "xor_distance",
]
