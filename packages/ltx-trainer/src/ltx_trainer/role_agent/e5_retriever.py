"""E5 retriever for search-augmented QA (paper Table 2 setup)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _default_corpus() -> dict[str, list[str]]:
    from ltx_trainer.role_agent.envs.search_qa_env import SEARCH_QA_CORPUS

    return SEARCH_QA_CORPUS


def _token_overlap_score(query: str, key: str) -> float:
    q = set(query.lower().split())
    k = set(key.lower().split())
    if not q or not k:
        return 0.0
    return len(q & k) / len(q)


@dataclass
class E5Retriever:
    """Dense E5 when sentence-transformers is available; else token overlap."""

    model_name: str = "intfloat/e5-small-v2"
    backend: str = "overlap"
    _model: Any = field(default=None, repr=False)
    _corpus_keys: list[str] = field(default_factory=list)
    _corpus_vecs: Any = field(default=None, repr=False)

    @classmethod
    def from_env(cls, corpus: dict[str, list[str]] | None = None) -> E5Retriever:
        mode = os.environ.get("GOPEX_ROLE_AGENT_RETRIEVER", "auto").lower()
        corpus = corpus or _default_corpus()
        if mode in ("overlap", "token"):
            return cls(backend="overlap")
        retriever = cls()
        if mode == "e5" or mode == "auto":
            try:
                retriever._init_e5(corpus)
            except Exception:
                retriever.backend = "overlap"
        return retriever

    def _init_e5(self, corpus: dict[str, list[str]]) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self._corpus_keys = list(corpus.keys())
        texts = [corpus[k][0] for k in self._corpus_keys]
        self._corpus_vecs = self._model.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
        )
        self.backend = "e5"

    def search(self, query: str, corpus: dict[str, list[str]] | None = None, *, top_k: int = 1) -> list[str]:
        corpus = corpus or _default_corpus()
        if self.backend == "e5" and self._model is not None and self._corpus_vecs is not None:
            qvec = self._model.encode([f"query: {query}"], normalize_embeddings=True)[0]
            scores = (self._corpus_vecs @ qvec).tolist()
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
            return [corpus[self._corpus_keys[i]][0] for i in ranked]
        key = query.strip().lower()
        ranked_keys = sorted(corpus.keys(), key=lambda k: -_token_overlap_score(key, k))
        out: list[str] = []
        for k in ranked_keys[:top_k]:
            if _token_overlap_score(key, k) > 0 or k in key or key in k:
                out.append(corpus[k][0])
        if not out and ranked_keys:
            out.append(corpus[ranked_keys[0]][0])
        return out

    def probe(self) -> dict[str, Any]:
        hit = self.search("france capital", top_k=1)
        return {
            "backend": self.backend,
            "model_name": self.model_name if self.backend == "e5" else None,
            "sample_hit": hit[0] if hit else None,
        }
