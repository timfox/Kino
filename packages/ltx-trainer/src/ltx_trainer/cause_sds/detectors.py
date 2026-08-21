"""Cause-aware error detectors — §3.3–3.4."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class ErrorCause(str, Enum):
    COMPREHENSION = "comprehension"
    PERCEPTION = "perception"
    DELETION = "deletion"


class DistortionEvent(str, Enum):
    CLEAN = "clean"
    INTERFERENCE = "interference"
    NOISE = "noise"
    RIR = "rir"
    PACKET_LOSS = "packet_loss"
    MISSING = "missing"


def predict_binary(embedding: np.ndarray, rng: np.random.Generator) -> bool:
    """Toy 1D-CNN head: error if norm of embedding perturbation is high."""
    e = np.asarray(embedding, dtype=np.float64).ravel()
    score = float(np.linalg.norm(e) + 0.1 * rng.standard_normal())
    return score > np.median(e) + 0.5 * np.std(e)


def deletion_mask_frame(
    predicted_token: str,
    no_emission: bool,
) -> bool:
    """et = I(ŷt = Del) · I(no emission) — Eq. (5)."""
    return predicted_token.lower() == "del" and no_emission


def fuse_error_causes(
    comprehension: bool,
    perception: bool,
    deletion: bool,
) -> ErrorCause | None:
    """Priority: Comprehension > Perception > Deletion — §3.5."""
    if comprehension:
        return ErrorCause.COMPREHENSION
    if perception:
        return ErrorCause.PERCEPTION
    if deletion:
        return ErrorCause.DELETION
    return None


def tag_span(cause: ErrorCause | None, distortion: DistortionEvent) -> str:
    if cause == ErrorCause.PERCEPTION:
        return "<noise>"
    if cause == ErrorCause.COMPREHENSION:
        return "<unknown>"
    if cause == ErrorCause.DELETION:
        return "<del>"
    if distortion == DistortionEvent.MISSING:
        return "<del>"
    return ""


def detector_profile(
    comp_flags: list[bool],
    perc_flags: list[bool],
    del_flags: list[bool],
    events: list[DistortionEvent],
) -> dict[str, Any]:
    return {
        "Y_comp": comp_flags,
        "Y_perc": perc_flags,
        "Y_del": del_flags,
        "Y_event": [e.value for e in events],
    }
