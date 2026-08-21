"""Structured-process checks (block-oriented safety vs arbitrary BPMN graphs)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cuta4bpm.metamodel import Block, SimpleActivity


def is_structured_element(el: Any) -> bool:
    if isinstance(el, (SimpleActivity, Block)):
        if isinstance(el, Block):
            return all(is_structured_element(c) for c in el.children)
        return True
    return False


def block_depth(el: Any) -> int:
    if isinstance(el, SimpleActivity):
        return 0
    if isinstance(el, Block):
        if not el.children:
            return 1
        return 1 + max(block_depth(c) for c in el.children)
    return 0


def count_activities(el: Any) -> int:
    if isinstance(el, SimpleActivity):
        return 1
    if isinstance(el, Block):
        return sum(count_activities(c) for c in el.children)
    return 0
