"""Inference-time skill mounting and prompt accounting (§3.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.latentskill.compiler import SkillCompiler
from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.lora import LatentSkillAdapter
from ltx_trainer.latentskill.skills import skill_document


@dataclass
class AgentTurnContext:
    """Single decision step: history only (latent) or history + skill text (in-context)."""

    history: str
    skill_name: str
    mode: str = "latent"  # latent | in_context
    injection_alpha: float = 0.6
    adapter: LatentSkillAdapter | None = None
    skill_text_in_prompt: bool = False

    def build_prompt(self) -> str:
        parts = [self.history]
        if self.mode == "in_context":
            parts.insert(0, skill_document(self.skill_name))
            self.skill_text_in_prompt = True
        else:
            self.skill_text_in_prompt = False
        return "\n\n".join(parts)

    def estimate_prefill_tokens(self, *, tokens_per_word: float = 1.3) -> float:
        prompt = self.build_prompt()
        return len(prompt.split()) * tokens_per_word


@dataclass
class LatentSkillSession:
    cfg: LatentSkillConfig = field(default_factory=LatentSkillConfig)
    compiler: SkillCompiler = field(default_factory=SkillCompiler)
    cache: dict[str, LatentSkillAdapter] = field(default_factory=dict)
    mounted: LatentSkillAdapter | None = None
    injection_alpha: float = 0.6

    def load_skill(self, skill_name: str) -> LatentSkillAdapter:
        sid = skill_name.replace(" ", "_").lower()
        if sid not in self.cache:
            self.cache[sid] = self.compiler.compile(skill_document(skill_name), sid)
        self.mounted = self.cache[sid]
        return self.mounted

    def turn(self, history: str, skill_name: str, *, mode: str = "latent") -> AgentTurnContext:
        adapter = self.load_skill(skill_name) if mode == "latent" else None
        return AgentTurnContext(
            history=history,
            skill_name=skill_name,
            mode=mode,
            injection_alpha=self.injection_alpha,
            adapter=adapter,
        )


# Keyword bias from compiled skill (stub policy prior for replay envs)
_SKILL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Pick And Place Skill": ("take", "put", "go to", "open"),
    "Look At Obj In Light Skill": ("desklamp", "lamp", "look", "examine"),
    "Clean Skill": ("clean", "sinkbasin", "fridge", "put"),
    "direct_retrieval": ("search", "answer", "click"),
    "multi_hop_reasoning": ("search", "retrieve", "therefore"),
}


def score_action_for_skill(action: str, skill_name: str) -> float:
    action_l = action.lower()
    keys = _SKILL_KEYWORDS.get(skill_name, ())
    return float(sum(1 for k in keys if k in action_l))


def choose_search_qa_action(valid_actions: list[str], *, step: int, mode: str = "latent") -> str:
    """Follow task hint plan (Search-QA fixtures); latent skips skill text in prefill."""
    concrete = [a for a in valid_actions if a.startswith("search[") or a.startswith("answer[")]
    if not concrete:
        return valid_actions[0] if valid_actions else "noop"
    idx = min(max(step - 1, 0), len(concrete) - 1)
    return concrete[idx]


def alpha_sweep_report(split: str = "seen") -> dict[str, Any]:
    from ltx_trainer.latentskill.analysis import alpha_performance_curve
    from ltx_trainer.latentskill.benchmarks import table11_alpha_sweep

    tab = table11_alpha_sweep()
    curve = alpha_performance_curve(split)
    key = "seen_avg_by_alpha" if split == "seen" else "unseen_avg_by_alpha"
    raw = tab[key]
    peak_alpha = max(raw, key=raw.get)
    return {
        "split": split,
        "curve": curve,
        "peak_alpha": peak_alpha,
        "peak_score": raw[peak_alpha],
        "optimal_paper": tab["optimal_alpha_seen" if split == "seen" else "optimal_alpha_unseen"],
        "inverted_u": raw.get(0.6, 0) > raw.get(1.2, 0) if split == "seen" else raw.get(0.5, 0) > raw.get(1.2, 0),
    }


def choose_action_stub(state: str, valid_actions: list[str], skill_name: str, *, mode: str = "latent") -> str:
    """Pick highest skill-keyword overlap; latent mode gets +0.5 bias (paper: better skill adherence)."""
    if not valid_actions:
        return "noop"
    if mode == "vanilla":
        return valid_actions[0]
    best = valid_actions[0]
    best_score = -1.0
    bias = 0.5 if mode == "latent" else 0.0
    for a in valid_actions:
        s = score_action_for_skill(a, skill_name) + bias
        if s > best_score:
            best_score = s
            best = a
    return best


def inference_notes(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    return {
        "backbone": cfg.backbone,
        "mount": "PEFT load generated LoRA on attn_o + mlp_down (preferred) or full 7 modules",
        "cache": "Compile each skill once → C[k] = G_phi(s_k); reuse across steps",
        "alpha": "Scale delta W by injection coefficient; α=0 recovers frozen backbone",
        "composition": "Sum adapters in weight space when components aligned (Eq. 6)",
        "vllm_hook": "Optional sidecar: compile skill → export safetensors → vLLM LoRA request",
        "messenger": "gopex_local_tools latentskill_compile + role_agent replay for smoke",
    }
