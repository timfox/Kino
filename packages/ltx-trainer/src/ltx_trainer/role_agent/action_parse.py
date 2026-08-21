"""Parse LLM agent responses into executable environment actions."""

from __future__ import annotations

import re


def parse_search_qa_response(text: str) -> str:
    """Parse Figure 7 <search> or <answer> into env action strings."""
    blob = text.strip()
    for tag in ("search", "answer"):
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", blob, re.DOTALL | re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            return f"{tag}[{content}]"
    m = re.search(r"^(search|answer)\[(.+)\]$", blob, re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}[{m.group(2).strip()}]"
    return blob


def parse_action_from_response(text: str, valid_actions: list[str], *, domain: str = "") -> str:
    """Map free-form LLM output to one valid graph action (case-insensitive)."""
    if domain.lower() == "search_qa":
        parsed = parse_search_qa_response(text)
        if parsed.startswith("search[") or parsed.startswith("answer["):
            return parsed
    if not valid_actions:
        return "noop"
    normalized = {a.strip().lower(): a for a in valid_actions}
    blob = text.strip()
    for tag in ("action", "response", "command"):
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", blob, re.DOTALL | re.IGNORECASE)
        if m:
            blob = m.group(1).strip()
            break
    blob = blob.strip().strip('"').strip("'")
    key = blob.lower()
    if key in normalized:
        return normalized[key]
    for line in blob.splitlines():
        candidate = line.strip().strip("-*").strip('"').strip("'").lower()
        if candidate in normalized:
            return normalized[candidate]
    lower_blob = blob.lower()
    for action in sorted(valid_actions, key=len, reverse=True):
        if action.lower() in lower_blob:
            return action
    return valid_actions[0]


def build_agent_prompt(
    *,
    domain: str,
    task: str,
    state: str,
    valid_actions: list[str],
) -> str:
    from ltx_trainer.role_agent.prompts import (
        ALFWORLD_AGENT_TEMPLATE,
        SEARCH_AGENT_TEMPLATE,
        TOY_AGENT_TEMPLATE,
        WEBSHOP_AGENT_TEMPLATE,
    )

    action_list = "\n".join(f"- {a}" for a in valid_actions)
    dom = domain.lower()
    if dom == "search_qa":
        step_count = state.count("<search>") if "<search>" in state else 0
        memory = state if "History:" in state else "(empty)"
        if memory.startswith("History:"):
            memory = memory.replace("History:", "", 1).strip()
        return SEARCH_AGENT_TEMPLATE.format(
            task_description=task,
            step_count=max(step_count, 0),
            memory_context=memory or "(empty)",
        )
    if dom == "alfworld":
        return ALFWORLD_AGENT_TEMPLATE.format(task=task, state=state, action_list=action_list)
    if dom == "webshop":
        return WEBSHOP_AGENT_TEMPLATE.format(task=task, state=state, action_list=action_list)
    return TOY_AGENT_TEMPLATE.format(task=task, state=state, action_list=action_list)
