"""Statistical and categorical phoneme confusion graphs (Sec. 2.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Minimal ARPABET subset for CPU demos
DEFAULT_PHONEMES: tuple[str, ...] = (
    "dh",
    "v",
    "z",
    "d",
    "t",
    "s",
    "th",
    "b",
    "ae",
    "ih",
    "iy",
    "ey",
    "ah",
    "ay",
)


@dataclass(frozen=True)
class ConfusionGraph:
    phonemes: tuple[str, ...]
    adjacency: np.ndarray  # shape (N, N), w[j_idx, i_idx] = P(realize i as j | error on i)

    def edge_weight(self, produced: str, target: str) -> float:
        i = self.phonemes.index(target)
        j = self.phonemes.index(produced)
        return float(self.adjacency[j, i])

    def outgoing_sum(self, target: str) -> float:
        i = self.phonemes.index(target)
        return float(self.adjacency[:, i].sum())


def build_statistical_graph(
    substitution_counts: dict[tuple[str, str], int],
    phonemes: Iterable[str] | None = None,
) -> ConfusionGraph:
    """Build directed weighted graph; Eq. (1) normalizes outgoing error edges per target."""
    vocab = tuple(phonemes) if phonemes is not None else DEFAULT_PHONEMES
    idx = {p: n for n, p in enumerate(vocab)}
    n = len(vocab)
    adj = np.zeros((n, n), dtype=np.float64)

    for (target, produced), count in substitution_counts.items():
        if target not in idx or produced not in idx or target == produced:
            continue
        adj[idx[produced], idx[target]] += float(count)

    for i, target in enumerate(vocab):
        denom = adj[:, i].sum()
        if denom > 0:
            adj[:, i] /= denom

    return ConfusionGraph(phonemes=vocab, adjacency=adj)


def build_categorical_graph(
    categories: dict[str, str],
    phonemes: Iterable[str] | None = None,
) -> ConfusionGraph:
    """Undirected uniform categorical graph (Yan et al. baseline)."""
    vocab = tuple(phonemes) if phonemes is not None else DEFAULT_PHONEMES
    n = len(vocab)
    adj = np.zeros((n, n), dtype=np.float64)
    by_cat: dict[str, list[int]] = {}
    for k, p in enumerate(vocab):
        cat = categories.get(p, "OTHER")
        by_cat.setdefault(cat, []).append(k)

    for members in by_cat.values():
        if len(members) < 2:
            continue
        w = 1.0 / (len(members) - 1)
        for a in members:
            for b in members:
                if a != b:
                    adj[a, b] = max(adj[a, b], w)
                    adj[b, a] = max(adj[b, a], w)
    return ConfusionGraph(phonemes=vocab, adjacency=adj)


def demo_substitution_counts_vietnamese() -> dict[tuple[str, str], int]:
    """Paper example: Vietnamese learners often substitute /z/ → /s/, rarely reverse."""
    return {
        ("z", "s"): 120,
        ("z", "d"): 10,
        ("s", "z"): 8,
        ("s", "t"): 52,
        ("dh", "d"): 90,
        ("dh", "t"): 5,
        ("th", "t"): 75,
        ("th", "s"): 8,
        ("v", "b"): 40,
        ("d", "dh"): 12,
        ("t", "th"): 10,
    }


def demo_substitution_counts_spanish() -> dict[tuple[str, str], int]:
    """Fig. 4 pairs: /d/-/dh/, /t/-/th/, /v/-/b/, /s/-/z/."""
    return {
        ("dh", "d"): 100,
        ("th", "t"): 95,
        ("v", "b"): 80,
        ("z", "s"): 70,
        ("s", "z"): 15,
        ("ih", "iy"): 55,
        ("iy", "ih"): 48,
    }


ARTICULATORY_CATEGORIES: dict[str, str] = {
    "p": "STOP",
    "b": "STOP",
    "t": "STOP",
    "d": "STOP",
    "k": "STOP",
    "g": "STOP",
    "s": "FRICATIVE",
    "z": "FRICATIVE",
    "sh": "FRICATIVE",
    "zh": "FRICATIVE",
    "dh": "FRICATIVE",
    "th": "FRICATIVE",
    "f": "FRICATIVE",
    "v": "FRICATIVE",
    "ih": "VOWEL",
    "iy": "VOWEL",
    "ey": "VOWEL",
    "ae": "VOWEL",
    "ah": "VOWEL",
    "ay": "VOWEL",
}
