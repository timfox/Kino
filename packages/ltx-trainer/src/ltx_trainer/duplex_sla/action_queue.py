"""Toy per-chunk action token spill (§2.3, §3.3) — splits a flat token list into ≤K-token chunks."""

from __future__ import annotations


def format_tool_call(function: str, arguments: str) -> str:
    """Wrap a function call in paper-style markers (§2.2)."""
    return (
        f'<|toolcall_begin|>{{"function": "{function}", "arguments": "{arguments}"}}'
        "<|toolcall_end|>"
    )


def spill_action_tokens(tokens: list[str], max_per_chunk: int = 10) -> list[list[str]]:
    """Partition tokens into chunk-aligned segments without splitting individual strings."""
    if max_per_chunk < 1:
        raise ValueError("max_per_chunk must be >= 1")
    out: list[list[str]] = []
    cur: list[str] = []
    for t in tokens:
        if len(cur) >= max_per_chunk:
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out
