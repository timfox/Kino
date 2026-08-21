"""Row-id normalization for frontier-guarded NRP (Sec. 5.2, Thm 5.14)."""

from __future__ import annotations

from ltx_trainer.nrp.embeddings import EDatabase, EFact


def ordered_row_id_expansion(db: EDatabase, order: tuple[str, ...] | None = None) -> EDatabase:
    """Build bD from e-database D (Sec. 5.2): Rdata + Remb + linear order R<."""
    expanded = EDatabase()
    row_ids: list[str] = []
    counter = 0
    for fact in db:
        rel, content, emb = fact.relation, fact.content, fact.embedding
        if rel.endswith("data") or rel.endswith("emb") or rel == "R<":
            continue
        row_id = f"r{counter}"
        counter += 1
        row_ids.append(row_id)
        expanded.add(EFact(f"{rel}data", (row_id,) + content, ()))
        if emb:
            expanded.add(EFact(f"{rel}emb", (row_id,), emb))
    ids = order if order is not None else tuple(sorted(row_ids))
    for i in range(len(ids) - 1):
        expanded.add(EFact("R<", (ids[i], ids[i + 1]), ()))
    return expanded


def is_row_id_normalized(db: EDatabase) -> bool:
    """Check schema shape: row-id first column; non-monadic EDBs have dim 0."""
    for fact in db:
        if fact.relation.endswith("data") and len(fact.content) >= 1:
            rid = fact.content[0]
            siblings = [f for f in db if f.relation.endswith("data") and f.content[0] == rid]
            if len(siblings) > 1:
                return False
    return True
