"""TrajGenAgent configuration (Li et al., arXiv:2606.12657)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.12657"
PAPER_TITLE = "TrajGenAgent: A Hierarchical LLM Agent for Human Mobility Trajectory Generation"
PAPER_AUTHORS = "Siyu Li, Toan Tran, Lingyi Zhao, Khurram Shafique, Li Xiong"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_REPO = "https://github.com/Emory-AIMS/TrajGenAgent"
BENCHMARK_NUMOSIM = "NumoSim — LA synthetic mobility (8 weeks, 200k individuals upstream)"
BENCHMARK_MOBILITYSYN = "MobilitySyn — week-long metropolitan simulation (5k individuals upstream)"

ACTIVITY_VOCABULARY = (
    "Home",
    "Work",
    "EatOut",
    "Shop",
    "Leisure",
    "Errand",
    "Education",
    "Healthcare",
)

NUMOSIM_ACTIVITIES = (
    "Home",
    "Work",
    "EatOut",
    "Shop",
    "Leisure",
    "Errand",
    "Education",
    "Healthcare",
    "Gym",
    "Social",
    "Transit",
    "Park",
    "Religious",
    "Care",
    "Other",
    "Nightlife",
)


@dataclass
class TrajGenAgentConfig:
    backbone_llm: str = "Qwen2.5-32B-Instruct"
    llm_temperature: float = 0.90
    llm_top_p: float = 0.95
    llm_max_tokens: int = 1024
    llm_max_context: int = 8192
    n_training_trajectories: int = 34_000
    n_individuals: int = 1_200
    peer_top_k: int = 5
    exploration_alpha: float = 0.15
    score_lambda_freq: float = 0.6
    score_lambda_dist: float = 0.4
    distance_beta: float = 0.02
    travel_min_minutes: float = 5.0
    travel_max_minutes: float = 180.0
    duration_min_minutes: float = 10.0
    duration_max_minutes: float = 240.0
    orchestrator_max_retries: int = 2
    use_kinematics: bool = True
    grid_km_numosim: float = 0.5
    grid_km_mobilitysyn: float = 0.7
    use_llm: bool = False
    vllm_base_url: str = "http://127.0.0.1:8002/v1"
    vllm_timeout_seconds: float = 120.0
    default_dataset: str = "NumoSim"
    reference_users: int = 24
