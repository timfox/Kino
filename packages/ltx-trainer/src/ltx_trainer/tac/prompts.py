"""Thinker / Answerer prompts (Appendix C, arXiv:2605.28713)."""

from __future__ import annotations

from ltx_trainer.tac.config import TaCConfig


def thinker_prompt(
    context: str,
    question: str,
    *,
    target_think_len: int,
    context_token_len: int,
    comp_ratio: int,
    cfg: TaCConfig | None = None,
) -> str:
    cfg = cfg or TaCConfig()
    return f"""You are a query-conditioned context compressor. Given the context and current information
need, write a high-quality {cfg.thinking_open} trace that preserves the context information a downstream
model needs to locate and use relevant information.

Task Guidelines:
1. Information Selection — extract query-relevant facts, entities, and relations; connect scattered evidence.
2. Think Requirements — keep the trace concise and structured; do NOT state the final answer.
3. Output Format:
{cfg.thinking_open}
[High-quality compressed context trace for downstream use.]
{cfg.thinking_close}

Context: {context}

Information Need: {question}

Compression Budget:
• Maximum thinking length: {target_think_len} tokens.
• Compression ratio: {comp_ratio}× from the original {context_token_len} context tokens.
"""


def answerer_prompt(question: str, thinking: str) -> str:
    return f"""Answer the question using ONLY the compressed thinking trace below (no other context).

Question: {question}

Compressed trace:
{thinking}

Give a short, direct answer:"""


def wrap_thinking(content: str, cfg: TaCConfig | None = None) -> str:
    cfg = cfg or TaCConfig()
    return f"{cfg.thinking_open}\n{content.strip()}\n{cfg.thinking_close}"
