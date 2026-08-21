"""NRP rule types and ReLU-FFN transformation stubs (Sec. 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

TransformFn = Callable[[tuple[float, ...]], tuple[float, ...]]
CombineFn = Callable[..., tuple[float, ...]]


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def relu_ffn(
    weights1: Sequence[Sequence[float]],
    bias1: Sequence[float],
    weights2: Sequence[Sequence[float]],
    bias2: Sequence[float],
) -> TransformFn:
    """Two-layer ReLU feed-forward network with rational parameters."""

    def _apply(x: tuple[float, ...]) -> tuple[float, ...]:
        hidden: list[float] = []
        for row, b in zip(weights1, bias1):
            s = b + sum(w * (x[i] if i < len(x) else 0.0) for i, w in enumerate(row))
            hidden.append(relu(s))
        out: list[float] = []
        for row, b in zip(weights2, bias2):
            s = b + sum(w * hidden[i] for i, w in enumerate(row))
            out.append(relu(s))
        return tuple(out)

    return _apply


def constant_transform(value: tuple[float, ...]) -> TransformFn:
    def _apply(_: tuple[float, ...]) -> tuple[float, ...]:
        return value

    return _apply


def scalar_indicator(divisor: float, accept_remainder_zero: bool = True) -> TransformFn:
    """Example 4.3: μ(x)=1 if x divisible by divisor else 0 (1-dim input)."""

    def _apply(x: tuple[float, ...]) -> tuple[float, ...]:
        v = x[0] if x else 0.0
        ok = abs(v % divisor) < 1e-9 if accept_remainder_zero else False
        return (1.0 if ok else 0.0,)

    return _apply


def select_dim(index: int) -> TransformFn:
    def _apply(x: tuple[float, ...]) -> tuple[float, ...]:
        return (x[index],) if index < len(x) else (0.0,)

    return _apply


def relu_threshold(threshold: float, out_dim: int = 1) -> TransformFn:
    """ReLU(1 - (e' - e)) style from Example 5.12."""

    def _apply(x: tuple[float, ...]) -> tuple[float, ...]:
        if len(x) < 2:
            return (0.0,) * out_dim
        return (relu(1.0 - (x[1] - x[0])),) + (0.0,) * (out_dim - 1)

    return _apply


@dataclass(frozen=True)
class Atom:
    relation: str
    variables: tuple[str, ...]


@dataclass(frozen=True)
class ConjunctionRule:
    head_relation: str
    head_vars: tuple[str, ...]
    body: tuple[Atom, ...]
    head_dim: int = 1
    combine_fn: CombineFn | None = None


@dataclass(frozen=True)
class DisjunctionRule:
    head_relation: str
    head_vars: tuple[str, ...]
    sources: tuple[str, ...]
    head_dim: int = 1


@dataclass(frozen=True)
class TransformationRule:
    head_relation: str
    head_vars: tuple[str, ...]
    source_relation: str
    transform: TransformFn
    head_dim: int = 1


Rule = ConjunctionRule | DisjunctionRule | TransformationRule
