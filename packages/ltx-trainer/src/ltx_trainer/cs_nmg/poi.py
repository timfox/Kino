"""POI detection and switch-boundary neighborhoods (PIER-style, §3)."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.cs_nmg.config import CsNmgConfig

_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def token_script_label(token: str) -> str:
    if _LATIN.search(token):
        return "latin"
    if _CJK.search(token):
        return "cjk"
    return "other"


def embedded_spans(tokens: list[str], *, matrix: str = "cjk") -> list[tuple[int, int]]:
    """Maximal contiguous segments whose script differs from matrix language."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(tokens):
        label = token_script_label(tokens[i])
        embedded = (matrix == "cjk" and label == "latin") or (matrix == "vie" and label == "latin")
        if embedded:
            start = i
            while i < len(tokens):
                lab = token_script_label(tokens[i])
                still = (matrix == "cjk" and lab == "latin") or (matrix == "vie" and lab == "latin")
                if not still:
                    break
                i += 1
            spans.append((start, i - 1))
        else:
            i += 1
    return spans


def poi_neighborhood(
    spans: list[tuple[int, int]],
    n_tokens: int,
    *,
    r: int = 1,
) -> list[int]:
    """Expand each embedded span by ±r tokens; merge duplicate indices."""
    idx: set[int] = set()
    for start, end in spans:
        lo = max(0, start - r)
        hi = min(n_tokens - 1, end + r)
        idx.update(range(lo, hi + 1))
    return sorted(idx)


def poi_index_set(reference: str, *, matrix: str = "cjk", r: int = 1) -> list[int]:
    tokens = reference.split()
    spans = embedded_spans(tokens, matrix=matrix)
    return poi_neighborhood(spans, len(tokens), r=r)


def pier(reference: str, hypothesis: str, *, poi_indices: list[int] | None = None) -> float:
    """Normalized Levenshtein on POI positions only (Eq. 12 stub)."""
    ref_toks = reference.split()
    hyp_toks = hypothesis.split()
    idx = poi_indices if poi_indices is not None else poi_index_set(reference)
    ref_poi = " ".join(ref_toks[i] for i in idx if i < len(ref_toks))
    hyp_poi = " ".join(hyp_toks[i] if i < len(hyp_toks) else "" for i in idx)
    return normalized_levenshtein(ref_poi, hyp_poi)


def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    dist = levenshtein_distance(a, b)
    denom = max(len(a), len(b), 1)
    return dist / denom


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def poi_demo(*, cfg: CsNmgConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CsNmgConfig()
    ref = "enzyme 5 alpha reductase được tạo ra"
    wrong = "enzyme 5 alpha reduc tây giờ được tạo ra"
    idx = poi_index_set(ref, matrix="vie", r=cfg.poi_neighborhood_r)
    return {
        "reference": ref,
        "poi_indices": idx,
        "wrong_hypothesis_pier": pier(ref, wrong, poi_indices=idx),
        "correct_hypothesis_pier": pier(ref, ref, poi_indices=idx),
        "errors_cluster_at_poi": pier(ref, wrong, poi_indices=idx) > 0.0,
    }
