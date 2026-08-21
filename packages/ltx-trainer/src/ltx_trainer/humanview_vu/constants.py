"""Human-view video MLLM survey constants (Meng et al., arXiv:2606.07433)."""

from __future__ import annotations

PAPER_ARXIV = "2606.07433"
PAPER_TITLE = "Watch, Remember, Reason: Human-View Video Understanding with MLLMs"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
AWESOME_URL = "https://github.com/marinero4972/Awesome-HumanView-VideoUnderstanding"

# Table 1 scope axes (survey comparison)
TABLE1_AXES = (
    "TG&SG",  # temporal and spatial grounding
    "Cap",  # video captioning
    "Omni",  # vision + audio + language
    "Efficiency",
    "Off-Mem",  # offline memory
    "Streaming-Mem",
    "Text-R",  # textual reasoning
    "O3-R",  # thinking-with-videos / o3-like reasoning
    "Subfields",
    "Train-Data",
    "Bench",
)
