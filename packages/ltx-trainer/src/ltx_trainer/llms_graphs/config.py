"""LLMs+Graphs tutorial configuration (Khan et al., arXiv:2606.11560)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.11560"
PAPER_TITLE = "LLMs+Graphs: Toward Graph-Native, Synergistic AI Systems"
PAPER_AUTHORS = "Arijit Khan, Longxu Sun, Xin Huang"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_REPO = "https://arxiv.org/abs/2606.11560"
PAPER_VENUE = "PAKDD Tutorial (Jun 2026)"

# Tutorial outline (Sec. 3) — minutes per section
TUTORIAL_SECTIONS: list[tuple[str, int]] = [
    ("Introduction", 30),
    ("LLMs for Graphs", 20),
    ("Graphs for LLMs", 20),
    ("Knowledge Graphs for LLMs", 20),
    ("LLMs for Knowledge Graphs", 20),
    ("Graphs for AI Agents", 20),
    ("AI Agents for Graphs", 20),
    ("Future Directions", 30),
]

SYNERGY_AXES = (
    "llms_for_graphs",
    "graphs_for_llms",
    "kgs_for_llms",
    "llms_for_kgs",
    "graphs_for_agents",
    "agents_for_graphs",
)


@dataclass
class LLMsGraphsConfig:
    tutorial_minutes: int = 180
    graph_rag_community_hops: int = 2
    kg_retrieval_top_k: int = 10
    agent_max_tool_steps: int = 8
