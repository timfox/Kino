"""Role-Agent — dual-role WIA + AIW agent RL (arXiv:2606.10917)."""

from ltx_trainer.role_agent.aiw import FailureMemory, FailureReflection, parse_reflection, parse_selected_tasks
from ltx_trainer.role_agent.config import RoleAgentConfig
from ltx_trainer.role_agent.gigpo import mixed_advantages
from ltx_trainer.role_agent.lms_similarity import lms_similarity
from ltx_trainer.role_agent.pipeline import evaluation_demo, evaluation_smoke, framework_card, knowledge_card
from ltx_trainer.role_agent.run_plan import run_plan
from ltx_trainer.role_agent.eval import run_llm_rollout_demo
from ltx_trainer.role_agent.training import run_toy_training
from ltx_trainer.role_agent.wia import compute_step_returns

__all__ = [
    "FailureMemory",
    "FailureReflection",
    "RoleAgentConfig",
    "compute_step_returns",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
    "lms_similarity",
    "mixed_advantages",
    "parse_reflection",
    "parse_selected_tasks",
    "run_llm_rollout_demo",
    "run_plan",
    "run_toy_training",
]
