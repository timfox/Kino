"""Tree measures μ(Tw) = 2^{-|w|} (Sec. 5)."""

from __future__ import annotations

from ltx_trainer.tree_eps_accept.trees import LabelledTree


def path_depth(w: str) -> int:
    return len(w.split(".")) if w else 0


def _is_prefix(w: str, v: str) -> bool:
    if not w:
        return True
    return v == w or v.startswith(f"{w}.")


def relative_measure(w: str, v: str) -> float:
    """μ(v | w) = 2^{|w|-|v|} when w is a prefix of v, else 0."""
    if not _is_prefix(w, v):
        return 0.0
    return float(2 ** (path_depth(w) - path_depth(v)))


def leaf_mass_from(w: str, tree: LabelledTree) -> float:
    """Σ_{l leaf, w≤l} μ(l | w) — Prop. 5.1 quantity."""
    return sum(relative_measure(w, leaf) for leaf in tree.leaves() if relative_measure(w, leaf) > 0)


def defect_ok_mass(tree: LabelledTree, *, ok_label: str = "ok", error_label: str = "error") -> float:
    """μ(dTerr) — mass of first-error-root subtrees (Prop. 5.3)."""
    seen: set[str] = set()
    total = 0.0
    for w in tree.paths():
        if tree.labels[w] != error_label:
            continue
        parent = w.rsplit(".", 1)[0] if "." in w else ""
        ok_chain = True
        cur = parent
        while cur:
            if tree.labels.get(cur) != ok_label:
                ok_chain = False
                break
            cur = cur.rsplit(".", 1)[0] if "." in cur else ""
        if not ok_chain or w in seen:
            continue
        # first error along path: parent must be ok or root error
        if parent and tree.labels.get(parent) == error_label:
            continue
        seen.add(w)
        total += relative_measure("", w)
    return total
