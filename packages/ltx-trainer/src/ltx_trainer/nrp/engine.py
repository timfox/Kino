"""NRP interpreter: conjunction / disjunction / transformation semantics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from ltx_trainer.nrp.embeddings import (
    EDatabase,
    EFact,
    product_combine,
    sum_aggregate,
    zero_vector,
)
from ltx_trainer.nrp.rules import (
    Atom,
    CombineFn,
    ConjunctionRule,
    DisjunctionRule,
    Rule,
    TransformationRule,
)


@dataclass(frozen=True)
class NRP:
    """Non-recursive sequence of rules (Sec. 3)."""

    rules: tuple[Rule, ...]
    output_relation: str | None = None

    @property
    def answer_relation(self) -> str:
        if self.output_relation:
            return self.output_relation
        last = self.rules[-1]
        return last.head_relation


def _head_key(rule: Rule) -> tuple[str, tuple[str, ...]]:
    if isinstance(rule, ConjunctionRule):
        return rule.head_relation, rule.head_vars
    if isinstance(rule, DisjunctionRule):
        return rule.head_relation, rule.head_vars
    return rule.head_relation, rule.head_vars


def _bindings_for_atom(db: EDatabase, atom: Atom) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    n = len(atom.variables)
    for fact in db.relation_facts(atom.relation):
        if len(fact.content) != n:
            continue
        results.append(dict(zip(atom.variables, fact.content)))
    return results


def _merge_bindings(
    left: dict[str, str], right: dict[str, str]
) -> dict[str, str] | None:
    merged = dict(left)
    for k, v in right.items():
        if k in merged and merged[k] != v:
            return None
        merged[k] = v
    return merged


def _join_body(
    db: EDatabase,
    body: tuple[Atom, ...],
    combine_fn: CombineFn | None = None,
) -> list[tuple[dict[str, str], list[tuple[float, ...]]]]:
    combine = combine_fn or product_combine
    if not body:
        return [({}, [()])]
    partials = _join_body(db, body[1:], combine_fn=combine_fn)
    first = body[0]
    out: list[tuple[dict[str, str], list[tuple[float, ...]]]] = []
    for bind in _bindings_for_atom(db, first):
        emb = db.get(first.relation, tuple(bind[v] for v in first.variables))
        embs = [emb] if emb is not None else [zero_vector(0)]
        for rest_bind, rest_embs in partials:
            merged = _merge_bindings(bind, rest_bind)
            if merged is None:
                continue
            combined_embs = [combine(e, re) for e in embs for re in rest_embs]
            out.append((merged, combined_embs))
    return out


def _apply_conjunction(db: EDatabase, rule: ConjunctionRule) -> EDatabase:
    out = EDatabase()
    grouped: dict[tuple[str, ...], list[tuple[float, ...]]] = defaultdict(list)
    for binding, embs in _join_body(db, rule.body, combine_fn=rule.combine_fn):
        head_content = tuple(binding[v] for v in rule.head_vars)
        for emb in embs:
            grouped[head_content].append(emb)
    dim = rule.head_dim
    for content, embs in grouped.items():
        agg = sum_aggregate(embs)
        if len(agg) < dim:
            agg = agg + zero_vector(dim - len(agg))
        elif len(agg) > dim:
            agg = agg[:dim]
        out.add(EFact(rule.head_relation, content, agg))
    return out


def _apply_disjunction(db: EDatabase, rule: DisjunctionRule) -> EDatabase:
    out = EDatabase()
    grouped: dict[tuple[str, ...], list[tuple[float, ...]]] = defaultdict(list)
    for src in rule.sources:
        for fact in db.relation_facts(src):
            if len(fact.content) != len(rule.head_vars):
                continue
            grouped[fact.content].append(fact.embedding)
    dim = rule.head_dim
    for content, embs in grouped.items():
        agg = sum_aggregate(embs)
        if len(agg) < dim:
            agg = agg + zero_vector(dim - len(agg))
        out.add(EFact(rule.head_relation, content, agg))
    return out


def _apply_transformation(db: EDatabase, rule: TransformationRule) -> EDatabase:
    out = EDatabase()
    for fact in db.relation_facts(rule.source_relation):
        if len(fact.content) != len(rule.head_vars):
            continue
        emb = rule.transform(fact.embedding)
        if len(emb) < rule.head_dim:
            emb = emb + zero_vector(rule.head_dim - len(emb))
        out.add(EFact(rule.head_relation, fact.content, emb))
    return out


def apply_rule(db: EDatabase, rule: Rule) -> EDatabase:
    if isinstance(rule, ConjunctionRule):
        return _apply_conjunction(db, rule)
    if isinstance(rule, DisjunctionRule):
        return _apply_disjunction(db, rule)
    return _apply_transformation(db, rule)


def execute(db: EDatabase, program: NRP) -> EDatabase:
    state = db
    for rule in program.rules:
        derived = apply_rule(state, rule)
        state = state.merge(derived)
    return state


AcceptancePolicy = Callable[[tuple[float, ...]], bool]


def default_acceptance(emb: tuple[float, ...]) -> bool:
    """Fixed policy: min element strictly positive (Sec. 3)."""
    return bool(emb) and min(emb) > 0.0


def gated_query(
    db: EDatabase,
    program: NRP,
    policy: AcceptancePolicy = default_acceptance,
) -> set[tuple[str, ...]]:
    result = execute(db, program)
    rel = program.answer_relation
    accepted: set[tuple[str, ...]] = set()
    for fact in result.relation_facts(rel):
        if policy(fact.embedding):
            accepted.add(fact.content)
    return accepted


def embedded_query(db: EDatabase, program: NRP) -> dict[tuple[str, ...], tuple[float, ...]]:
    result = execute(db, program)
    rel = program.answer_relation
    return {f.content: f.embedding for f in result.relation_facts(rel)}
