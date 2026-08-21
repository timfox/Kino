"""Scope notes for R-FUML reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Reference implementation covers fuzzy credibility, RMF, Lccl, and RLVC scheduling — not full DNN training on eight datasets.",
    "View-conflict injection (V−2 mislabeled views + 10% Gaussian noise on test) is described in the paper but not simulated at scale here.",
    "GMM conflict division is a lightweight 1D EM stub; production code should use the paper's full pipeline and released weights upon acceptance.",
    "LTX integration: use RMF-style fusion when combining heterogeneous modality losses or embeddings with suspected cross-modal conflict.",
)
