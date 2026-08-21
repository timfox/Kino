"""Ablation: latent vs in-context vs vanilla (Tables 1–2, §4)."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.benchmarks import table1_alfworld, table2_search_qa
from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.rollout import run_alfworld_fixture_compare, run_search_qa_compare


def ablation_table_anchors() -> dict[str, Any]:
    t1 = table1_alfworld()
    t2 = table2_search_qa()
    return {
        "alfworld": {
            "LatentSkill": t1["LatentSkill"]["seen_avg"],
            "In-Context Skill": t1["In-Context Skill"]["seen_avg"],
            "Vanilla": t1["Vanilla"]["seen_avg"],
        },
        "search_qa": {
            "LatentSkill": t2["LatentSkill"]["avg"],
            "In-Context Skill": t2["In-Context Skill"]["avg"],
            "RAG": t2["RAG"]["avg"],
        },
    }


def run_ablation_smoke(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    """CPU replay ablation: latent should beat in-context on token cost; both beat vanilla on paper anchors."""
    cfg = cfg or LatentSkillConfig()
    alf = run_alfworld_fixture_compare()
    search = run_search_qa_compare()
    anchors = ablation_table_anchors()
    latent_alf_ok = alf["latent"]["success"]
    latent_search_ok = search["latent"]["success"]
    token_win_alf = alf["latent"]["prefill_k"] < alf["in_context"]["prefill_k"]
    token_win_search = search["latent"]["prefill_k"] < search["in_context"]["prefill_k"]
    paper_latent_beats_inctx = (
        anchors["alfworld"]["LatentSkill"] > anchors["alfworld"]["In-Context Skill"]
        and anchors["search_qa"]["LatentSkill"] > anchors["search_qa"]["In-Context Skill"]
    )
    return {
        "ok": latent_alf_ok and latent_search_ok and token_win_alf and token_win_search and paper_latent_beats_inctx,
        "paper_anchors": anchors,
        "alfworld_replay": {
            "latent": alf["latent"],
            "in_context": alf["in_context"],
            "token_reduction": alf["token_compare"]["relative_reduction"],
        },
        "search_qa_replay": {
            "latent": search["latent"],
            "in_context": search["in_context"],
            "token_reduction": search["token_compare"]["relative_reduction"],
        },
        "note": "Replay success is stub policy smoke; paper Table 1–2 numbers are upstream anchors.",
    }
