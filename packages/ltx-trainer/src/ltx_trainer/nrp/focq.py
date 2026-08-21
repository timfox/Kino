"""Minimal FOCQ interpreter stub (Sec. 6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WeightedStructure:
    """Real-weighted σ-structure over finite domain."""

    domain: tuple[str, ...]
    relations: dict[str, dict[tuple[str, ...], tuple[float, ...]]]

    def rel(self, name: str, args: tuple[str, ...]) -> tuple[float, ...]:
        if name not in self.relations:
            return (0.0,)
        return self.relations[name].get(args, (0.0,))


def from_edatabase(domain: tuple[str, ...], edb_facts: dict[tuple[str, tuple[str, ...]], tuple[float, ...]]) -> WeightedStructure:
    rels: dict[str, dict[tuple[str, ...], tuple[float, ...]]] = {}
    for (rel, content), emb in edb_facts.items():
        dim = len(emb)
        presence = rels.setdefault(rel, {})
        presence[content] = (1.0,) + emb if dim else (1.0,)
        for i in range(dim):
            key = f"{rel}{i + 1}"
            slot = rels.setdefault(key, {})
            slot[content] = (emb[i],)
    return WeightedStructure(domain=domain, relations=rels)


@dataclass(frozen=True)
class FocqTerm:
    kind: str
    payload: Any = None
    left: FocqTerm | None = None
    right: FocqTerm | None = None
    vars: tuple[str, ...] = ()


def q(value: float) -> FocqTerm:
    return FocqTerm("q", value)


def rel(name: str, vars: tuple[str, ...]) -> FocqTerm:
    return FocqTerm("rel", (name, vars))


def add(a: FocqTerm, b: FocqTerm) -> FocqTerm:
    return FocqTerm("add", left=a, right=b)


def mul(a: FocqTerm, b: FocqTerm) -> FocqTerm:
    return FocqTerm("mul", left=a, right=b)


def neg(a: FocqTerm) -> FocqTerm:
    return FocqTerm("neg", left=a)


def sum_term(vars: tuple[str, ...], body: FocqTerm) -> FocqTerm:
    return FocqTerm("sum", vars=vars, left=body)


def max_term(a: FocqTerm, b: FocqTerm) -> FocqTerm:
    return FocqTerm("max", left=a, right=b)


def gt(a: FocqTerm, b: FocqTerm) -> FocqTerm:
    return FocqTerm("gt", left=a, right=b)


def eval_term(term: FocqTerm, struct: WeightedStructure, binding: dict[str, str]) -> float:
    if term.kind == "q":
        return float(term.payload)
    if term.kind == "rel":
        name, vars = term.payload
        args = tuple(binding[v] for v in vars)
        vec = struct.rel(name, args)
        return vec[0] if vec else 0.0
    if term.kind == "add":
        assert term.left and term.right
        return eval_term(term.left, struct, binding) + eval_term(term.right, struct, binding)
    if term.kind == "mul":
        assert term.left and term.right
        return eval_term(term.left, struct, binding) * eval_term(term.right, struct, binding)
    if term.kind == "neg":
        assert term.left
        return -eval_term(term.left, struct, binding)
    if term.kind == "max":
        assert term.left and term.right
        return max(eval_term(term.left, struct, binding), eval_term(term.right, struct, binding))
    if term.kind == "sum":
        assert term.left
        total = 0.0
        free = [v for v in term.vars if v not in binding]
        if not free:
            return eval_term(term.left, struct, binding)
        _sum_over(free, 0, binding, struct, term.left, total_holder := [0.0])
        return total_holder[0]
    raise ValueError(f"unknown FOCQ term kind {term.kind}")


def _sum_over(
    free: list[str],
    idx: int,
    binding: dict[str, str],
    struct: WeightedStructure,
    body: FocqTerm,
    acc: list[float],
) -> None:
    if idx == len(free):
        acc[0] += eval_term(body, struct, binding)
        return
    var = free[idx]
    for el in struct.domain:
        binding[var] = el
        _sum_over(free, idx + 1, binding, struct, body, acc)


def eval_formula(formula: FocqTerm, struct: WeightedStructure, binding: dict[str, str]) -> bool:
    if formula.kind == "gt":
        assert formula.left and formula.right
        return eval_term(formula.left, struct, binding) > eval_term(formula.right, struct, binding)
    raise ValueError(f"unknown FOCQ formula kind {formula.kind}")
