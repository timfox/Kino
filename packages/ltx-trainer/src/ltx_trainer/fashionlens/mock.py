"""FashionLens PGSQC smoke."""

from __future__ import annotations

from typing import Any


def _unit_vector(dim: int, *, peak: int = 0) -> list[float]:
    v = [0.0] * dim
    v[peak % dim] = 1.0
    return v


def toy_unit_query(*, dim: int = 3) -> list[float]:
    return _unit_vector(dim, peak=0)


def toy_unit_target(*, dim: int = 3) -> list[float]:
    return _unit_vector(dim, peak=1 % dim)


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.fashionlens.pgsqc import adaptation_proposal, dot

    q = toy_unit_query()
    t = toy_unit_target()
    q_p = adaptation_proposal(q)
    return {"query_dim": len(q), "proposal_dot": round(dot(q, q_p), 4), "target_dot": round(dot(q, t), 4)}
