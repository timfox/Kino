"""First-order logic parsing, vocabulary, and verbalisation (arXiv:2606.05812)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class RefinementOperator(str, Enum):
    ATTR = "o_attr"
    REL = "o_rel"
    NEG = "o_neg"


# Representative subset of V_audio (642 total in paper)
V_AUDIO: tuple[str, ...] = (
    "Bird",
    "Chirping",
    "Morning",
    "Outdoor",
    "Peaceful",
    "DistressCall",
    "Shrieking",
    "AlarmCall",
    "Speaking",
    "Quiet",
    "Shouting",
    "Footsteps",
    "Empty",
    "Rain",
    "Falling",
    "MetalRoof",
    "Humming",
    "Machinery",
    "Interference",
    "Wind",
    "Blowing",
)


@dataclass
class FolForm:
    """Lightweight FOL form as a set of predicate strings."""

    predicates: set[str] = field(default_factory=set)
    negated: set[str] = field(default_factory=set)
    relations: list[tuple[str, str, str]] = field(default_factory=list)  # R, x, y
    raw: str = ""

    def copy(self) -> FolForm:
        return FolForm(
            predicates=set(self.predicates),
            negated=set(self.negated),
            relations=list(self.relations),
            raw=self.raw,
        )


def pred_set(phi: FolForm) -> set[str]:
    return set(phi.predicates) | {f"¬{p}" for p in phi.negated}


def invariant_set(phi0: FolForm, phi_plus: FolForm, phi_minus: FolForm) -> set[str]:
    c = pred_set(phi0) & pred_set(phi_plus) & pred_set(phi_minus)
    return c if c else pred_set(phi0)


def parse_query_fallback(query: str) -> FolForm:
    """Toy parser: keyword → predicates with fallback to root verb + object."""
    q = query.lower().strip()
    preds: set[str] = set()
    neg: set[str] = set()
    if "bird" in q and "chirp" in q:
        preds.update({"Bird", "Chirping"})
        if "morning" in q:
            preds.add("Morning")
    elif "talk" in q or "speak" in q:
        preds.add("Speaking")
        if "quiet" in q:
            preds.add("Quiet")
        if "without" in q and "shout" in q:
            neg.add("Shouting")
    elif "footstep" in q:
        preds.add("Footsteps")
        if "quiet" in q:
            preds.add("Quiet")
        if "empty" in q or "corridor" in q:
            preds.update({"Empty", "In"})
    elif "rain" in q and "metal" in q:
        preds.update({"Rain", "Falling", "MetalRoof", "On"})
    elif "machin" in q and "hum" in q:
        preds.update({"Machinery", "Humming"})
        if "electrical" in q or "interference" in q:
            preds.add("Interference")  # may be OOV in full vocab
    elif "wind" in q:
        preds.update({"Wind", "Blowing"})
        if "without rain" in q or "no rain" in q:
            neg.add("Rain")
    else:
        tokens = re.findall(r"[a-z]+", q)
        if len(tokens) >= 2:
            preds.add(tokens[0].capitalize() + tokens[1].capitalize())
        elif tokens:
            preds.add(tokens[0].capitalize())
    if not preds and not neg:
        root = re.findall(r"[a-z]+", q)
        event = "".join(w.capitalize() for w in root[:2]) or "Event"
        preds.add(f"RootEvent_{event}")
    raw = _serialize(preds, neg)
    return FolForm(predicates=preds, negated=neg, raw=raw)


def apply_operator(phi: FolForm, op: RefinementOperator, arg: str) -> FolForm | None:
    out = phi.copy()
    if op == RefinementOperator.ATTR:
        if arg not in out.predicates:
            out.predicates.add(arg)
    elif op == RefinementOperator.REL:
        out.relations.append((arg, "x", "y"))
        out.predicates.add(arg)
    elif op == RefinementOperator.NEG:
        if arg in out.predicates:
            out.predicates.discard(arg)
        out.negated.add(arg)
    else:
        return None
    out.raw = _serialize(out.predicates, out.negated)
    return out


def verbalise(phi: FolForm) -> str:
    """Template-based G(·) verbaliser stub."""
    parts: list[str] = []
    for p in sorted(phi.predicates):
        parts.append(p.lower())
    for n in sorted(phi.negated):
        parts.append(f"not {n.lower()}")
    if not parts:
        return "an audio event"
    text = ", ".join(parts)
    return f"a scene with {text}"


def _serialize(preds: set[str], neg: set[str]) -> str:
    pos = " ∧ ".join(sorted(preds)) if preds else ""
    neg_s = " ∧ ".join(f"¬{n}" for n in sorted(neg)) if neg else ""
    if pos and neg_s:
        return f"∃x ({pos} ∧ {neg_s})"
    return f"∃x ({pos or neg_s})"


def predicate_overlap(phi_a: FolForm, phi_b: FolForm) -> float:
    """Jaccard-like score normalised by geometric mean (Eq. 8 component)."""
    pa = pred_set(phi_a)
    pb = pred_set(phi_b)
    if not pa or not pb:
        return 0.0
    inter = len(pa & pb)
    return inter / (len(pa) * len(pb)) ** 0.5
