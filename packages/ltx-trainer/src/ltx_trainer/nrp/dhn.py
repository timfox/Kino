"""Deep Homomorphism Network stub (Defs 5.1–5.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ltx_trainer.nrp.embeddings import product_combine, sum_aggregate, zero_vector
from ltx_trainer.nrp.rules import TransformFn


@dataclass(frozen=True)
class ClassicalFact:
    relation: str
    content: tuple[str, ...]


@dataclass
class PointedDatabase:
    """Pointed classical database F• (Def 5.1)."""

    facts: set[tuple[str, tuple[str, ...]]]
    distinguished: str

    def add(self, relation: str, content: tuple[str, ...]) -> None:
        self.facts.add((relation, content))

    def active_domain(self) -> set[str]:
        dom: set[str] = set()
        for _, content in self.facts:
            dom.update(content)
        return dom


@dataclass
class ElementEmbeddedDatabase:
    """Element-embedded database (D, λ) (Sec. 5.1)."""

    database: PointedDatabase
    labels: dict[str, tuple[float, ...]]

    def lambda_at(self, element: str) -> tuple[float, ...]:
        return self.labels.get(element, zero_vector(1))


@dataclass(frozen=True)
class HomomorphismQuery:
    pattern: PointedDatabase
    transforms: dict[str, TransformFn]

    def output_dim(self) -> int:
        for fn in self.transforms.values():
            sample = fn((1.0,))
            if sample:
                return len(sample)
        return 1


def homomorphisms(pattern: PointedDatabase, target: PointedDatabase) -> list[dict[str, str]]:
    """Homomorphisms h with h(distinguished_pattern) = target.distinguished."""
    pat_dom = pattern.active_domain()
    tgt_dom = target.active_domain()
    if pattern.distinguished not in pat_dom or target.distinguished not in tgt_dom:
        return []

    def _preserved(h: dict[str, str]) -> bool:
        for rel, content in pattern.facts:
            mapped = tuple(h.get(c, c) for c in content)
            if (rel, mapped) not in target.facts:
                return False
        return h.get(pattern.distinguished, pattern.distinguished) == target.distinguished

    def _extend(partial: dict[str, str], remaining: list[str]) -> list[dict[str, str]]:
        if not remaining:
            return [dict(partial)] if _preserved(partial) else []
        var = remaining[0]
        if var in partial:
            return _extend(partial, remaining[1:])
        outs: list[dict[str, str]] = []
        for el in tgt_dom:
            outs.extend(_extend({**partial, var: el}, remaining[1:]))
        return outs

    vars_list = sorted(pat_dom)
    return _extend({}, vars_list)


def eval_homomorphism_query(
    query: HomomorphismQuery,
    pointed_target: ElementEmbeddedDatabase,
) -> tuple[float, ...]:
    """eval((F•, μ), (Da, λ)) from Def 5.1."""
    target = pointed_target.database
    maps = homomorphisms(query.pattern, target)
    vectors: list[tuple[float, ...]] = []
    for h in maps:
        parts: list[tuple[float, ...]] = []
        for y in query.pattern.active_domain():
            fn = query.transforms.get(y, lambda x: x)
            parts.append(fn(pointed_target.lambda_at(h[y])))
        if parts:
            vectors.append(product_combine(*parts))
    if not vectors:
        return zero_vector(query.output_dim())
    return sum_aggregate(vectors)


@dataclass(frozen=True)
class DHNLayer:
    queries: tuple[HomomorphismQuery, ...]
    rho: TransformFn


@dataclass(frozen=True)
class DHN:
    layers: tuple[DHNLayer, ...]


def apply_dhn_layer(
    layer: DHNLayer,
    db: ElementEmbeddedDatabase,
) -> ElementEmbeddedDatabase:
    new_labels: dict[str, tuple[float, ...]] = {}
    for element in db.database.active_domain():
        db.database.distinguished = element
        parts = [eval_homomorphism_query(q, db) for q in layer.queries]
        combined = product_combine(*parts) if parts else zero_vector(1)
        new_labels[element] = layer.rho(combined)
    return ElementEmbeddedDatabase(db.database, new_labels)


def apply_dhn(dhn: DHN, db: ElementEmbeddedDatabase) -> ElementEmbeddedDatabase:
    state = db
    for layer in dhn.layers:
        state = apply_dhn_layer(layer, state)
    return state


def triangle_homomorphism_query() -> HomomorphismQuery:
    """Example 5.6 DHN layer: triangle pattern with identity transforms."""
    pat = PointedDatabase(facts=set(), distinguished="u")
    for edge in [("u", "v1"), ("v1", "v2"), ("v2", "u")]:
        pat.add("E", edge)
        pat.add("E", (edge[1], edge[0]))
    for v in ["u", "v1", "v2"]:
        pat.add("V", (v,))
    transforms = {v: lambda x, _v=v: x for v in ["u", "v1", "v2"]}
    return HomomorphismQuery(pattern=pat, transforms=transforms)


def demo_dhn_triangle() -> dict[str, object]:
    db = PointedDatabase(facts=set(), distinguished="u")
    for a, b in [("u", "v1"), ("v1", "v2"), ("v2", "u"), ("u", "v3")]:
        db.add("E", (a, b))
        db.add("E", (b, a))
    for v, r in [("u", 2.0), ("v1", 3.0), ("v2", 1.0), ("v3", 4.0)]:
        db.add("V", (v,))
    labels = {v: (r,) for v, r in [("u", 2.0), ("v1", 3.0), ("v2", 1.0), ("v3", 4.0)]}
    eedb = ElementEmbeddedDatabase(db, labels)
    layer = DHNLayer(
        queries=(triangle_homomorphism_query(),),
        rho=lambda x: (sum(x) if x else 0.0,),
    )
    out = apply_dhn_layer(layer, eedb)
    return {"u_embedding": out.labels.get("u"), "labels": out.labels}
