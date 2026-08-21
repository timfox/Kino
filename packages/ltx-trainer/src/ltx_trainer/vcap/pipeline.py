"""VCap framework card, demos, and evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vcap.benchmarks import benchmarks_bundle
from ltx_trainer.vcap.config import VCapConfig
from ltx_trainer.vcap.hypergeometric import p_collision, p_recall
from ltx_trainer.vcap.rewards import grpo_advantages, mock_judge_scores, sentence_reward


def framework_card(cfg: VCapConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VCapConfig()
    bench = benchmarks_bundle()
    return {
        "name": "VCap — Witness-Adjudicator hypergeometric rewards",
        "paper": f"arXiv:{cfg.arxiv}",
        "roles": {
            "witness": "reference caption R = Φ(y_ref) — stochastic fact subset",
            "adjudicator": "visual signal x — verifies overlaps at armed slots",
        },
        "reward": {
            "image": "r = wcorr*scorr + wcomp*scomp + wtxt*stxt",
            "video": "r_global + w_local * r_local",
            "weights": {"wcorr": cfg.wcorr, "wcomp": cfg.wcomp, "wtxt": cfg.wtxt, "wlocal": cfg.wlocal_video},
        },
        "training": {
            "algorithm": "GRPO",
            "backbone": cfg.backbone_policy,
            "epochs": ["e1 weak reference pool", "e2 self-evolved references"],
        },
        "hypergeometric": {
            "p_recall": "∏ (c-i)/(N-i)",
            "p_collision": "∏ (N-m-j)/(N-j)",
            "optimum": "c→N, n-c→0 independent of witness size m",
        },
        "benchmarks": bench,
        "ltx_integration": {
            "dense_caption_instruction": "ltx_trainer.vcap.captioning",
            "preprocess_meta": "vcap_preprocess_extra / attach_vcap_to_preprocess_meta",
            "pair_scoring": "score_caption_pair (mock judge for QA)",
        },
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.28023",
        "title": "VCap: Hypergeometric Rewards for Weak-to-Strong Visual Captioning",
        "insight": "Reference is witness not imitation target; image adjudicates fact-level correctness and completeness",
        "results": {
            "capmas_avg_e2": 83.53,
            "decap_avg_e2": 73.67,
            "vdc_avg_e2": 36.01,
            "vs_gpt54_capmas": 81.36,
        },
    }


def evaluation_demo() -> dict[str, Any]:
    cfg = VCapConfig()
    ref = "A silver truck is parked by a beige building. Three people walk on the sidewalk. A flowering tree with pink blossoms dominates the foreground."
    weak = "A tree with flowers. People walk. A building and truck are visible."
    strong = (
        ref
        + " The tree has reddish-brown leaves mixed with blossoms. A red curb lines the street. "
        "HVAC units are mounted on the building wall. Pedestrians wear dark coats."
    )
    s_weak = mock_judge_scores(weak, ref, cfg=cfg)
    s_strong = mock_judge_scores(strong, ref, cfg=cfg)
    c_w, n_w, m = 8, 12, 10
    c_s, n_s, _ = 10, 12, 10
    rewards_rollout = [sentence_reward(mock_judge_scores(strong, ref, cfg=cfg), cfg) for _ in range(4)]
    rewards_rollout[2] = sentence_reward(s_weak, cfg)
    adv = grpo_advantages(rewards_rollout)
    return {
        "weak_reward": sentence_reward(s_weak, cfg),
        "strong_reward": sentence_reward(s_strong, cfg),
        "p_recall_weak": p_recall(c_w, m, cfg.latent_facts_N),
        "p_recall_strong": p_recall(c_s, m, cfg.latent_facts_N),
        "p_collision_weak": p_collision(max(0, n_w - c_w), m, cfg.latent_facts_N),
        "p_collision_strong": p_collision(max(0, n_s - c_s), m, cfg.latent_facts_N),
        "grpo_advantages_sample": adv,
        "improves_with_denser_policy": sentence_reward(s_strong, cfg) > sentence_reward(s_weak, cfg),
        "bench_e2_beats_gpt": benchmarks_bundle()["vcap_e2_capmas_avg"] > benchmarks_bundle()["gpt54_capmas_avg"],
    }


def evaluation_smoke(cfg: VCapConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VCapConfig()
    demo = evaluation_demo()
    ok = (
        demo["improves_with_denser_policy"]
        and demo["p_recall_strong"] >= demo["p_recall_weak"]
        and demo["bench_e2_beats_gpt"]
        and len(demo["grpo_advantages_sample"]) == 4
    )
    return {
        "paper": f"arXiv:{cfg.arxiv}",
        "ok": ok,
        "demo": demo,
    }
