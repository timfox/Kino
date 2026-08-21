"""Skill library, templates, and task matching (Appendix B, arXiv:2606.06087)."""
from __future__ import annotations

from typing import Any

# ALFWorld Table 5
ALFWORLD_SKILL_MAP: dict[str, str] = {
    "Pick": "Pick And Place Skill",
    "Pick2": "Pick And Place Skill",
    "Look": "Look At Obj In Light Skill",
    "Clean": "Clean Skill",
    "Heat": "Heat Skill",
    "Cool": "Cool Skill",
}

# Search-QA Table 6
SEARCH_QA_SKILL_MAP: dict[str, str] = {
    "NQ": "direct_retrieval",
    "TriviaQA": "direct_retrieval",
    "PopQA": "direct_retrieval",
    "Bamboogle": "multi_hop_reasoning",
    "MuSiQue": "multi_hop_reasoning",
    "HotpotQA": "multi_hop_reasoning",
    "2WikiMultihopQA": "multi_hop_reasoning",
}

SEARCH_QA_COMPARISON_SKILL: dict[str, str] = {
    "HotpotQA": "comparison",
    "2WikiMultihopQA": "comparison",
}

# Minimal markdown bodies for stub compilation (SkillRL-style structure)
SKILL_DOCUMENTS: dict[str, str] = {
    "Pick And Place Skill": """# Pick And Place
- Search drawers, shelves, and sidetables systematically.
- Take target object before navigating to receptacle.
- Apply when: pick and place household objects.
""",
    "Look At Obj In Light Skill": """# Look At Obj In Light
- Locate desklamp and target object; activate lamp while holding object.
- Apply when: examine object under light.
""",
    "Clean Skill": """# Clean
- Clean object at sinkbasin before placing in receptacle.
- Apply when: pick clean then place.
""",
    "Heat Skill": """# Heat
- Heat object in microwave before delivery.
""",
    "Cool Skill": """# Cool
- Cool object in fridge before delivery.
""",
    "direct_retrieval": """# Direct Retrieval
- Retrieve top passages; answer with short span.
""",
    "multi_hop_reasoning": """# Multi-hop Reasoning
- Decompose query; retrieve supporting facts sequentially.
""",
    "comparison": """# Comparison
- Retrieve facts for both entities; compare attribute.
""",
}

# Component decomposition for Look + Pick composition (§4.5, Appendix H)
SKILL_COMPONENTS: dict[str, list[str]] = {
    "Look At Obj In Light Skill": [
        "general navigation heuristics",
        "mistake avoidance patterns",
        "lamp interaction task-specific",
    ],
    "Pick And Place Skill": [
        "general navigation heuristics",
        "mistake avoidance patterns",
        "systematic object search task-specific",
    ],
}


def match_alfworld_skill(task_type: str) -> str:
    return ALFWORLD_SKILL_MAP[task_type]


def match_search_qa_skill(dataset: str, *, sub_type: str | None = None) -> str:
    if sub_type == "comparison" and dataset in SEARCH_QA_COMPARISON_SKILL:
        return SEARCH_QA_COMPARISON_SKILL[dataset]
    return SEARCH_QA_SKILL_MAP[dataset]


def skill_document(skill_name: str) -> str:
    return SKILL_DOCUMENTS.get(skill_name, f"# {skill_name}\n- Apply procedural heuristics.\n")


def decompose_skill_components(skill_name: str) -> list[str]:
    return list(SKILL_COMPONENTS.get(skill_name, [skill_document(skill_name)]))


def skill_library_manifest() -> dict[str, Any]:
    return {
        "alfworld": ALFWORLD_SKILL_MAP,
        "search_qa": SEARCH_QA_SKILL_MAP,
        "comparison_override": SEARCH_QA_COMPARISON_SKILL,
        "documents": list(SKILL_DOCUMENTS.keys()),
        "component_skills": list(SKILL_COMPONENTS.keys()),
        "source": "Xia et al. 2026 SkillRL library (stub templates)",
    }
