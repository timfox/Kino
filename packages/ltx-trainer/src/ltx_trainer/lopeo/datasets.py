"""Balanced / unbalanced trial construction — Section II-C."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lopeo.balance import balance_index, counts_from_trials


def construct_kul_trials(
    *,
    balanced: bool,
    trials_per_subject: int = 8,
    n_subjects: int = 2,
    speakers: tuple[str, str] = ("spk0", "spk1"),
) -> list[dict[str, Any]]:
    """KUL-style 2-speaker trials: BI≈0 (counterbalanced) or BI≈1 (one speaker always attended)."""
    a, b = speakers
    trials: list[dict[str, Any]] = []
    tid = 0
    for subj in range(n_subjects):
        for t in range(trials_per_subject):
            if balanced:
                if t % 2 == 0:
                    att, unatt = a, b
                else:
                    att, unatt = b, a
            else:
                att, unatt = a, b
            trials.append(
                {
                    "trial_id": f"kul_s{subj}_t{t}",
                    "subject": subj,
                    "attended": att,
                    "unattended": unatt,
                }
            )
            tid += 1
    return trials


def construct_dtu_trials(
    *,
    balanced: bool,
    n_trials: int = 12,
    attended_pool: tuple[str, ...] = ("aud0", "aud1", "aud2", "aud3"),
) -> list[dict[str, Any]]:
    """DTU-style: unbalanced retains only trials whose attended stream is in attended_pool."""
    all_streams = tuple(f"stream{i}" for i in range(len(attended_pool) * 2))
    trials: list[dict[str, Any]] = []
    for i in range(n_trials):
        att = all_streams[i % len(all_streams)]
        unatt = all_streams[(i + 1) % len(all_streams)]
        if balanced or att in attended_pool:
            trials.append(
                {
                    "trial_id": f"dtu_{i}",
                    "attended": att,
                    "unattended": unatt,
                }
            )
    return trials


def construct_nju_ceegrid_trials(
    *,
    balanced: bool,
    trials_per_speaker: int = 3,
    speakers: tuple[str, ...] = ("news_a", "news_b", "news_c"),
) -> list[dict[str, Any]]:
    """NJU cEEGrid 3-speaker trials; unbalanced keeps one attended speaker only (LOEO partition)."""
    trials: list[dict[str, Any]] = []
    tid = 0
    target_speakers = speakers if balanced else (speakers[0],)
    for att in target_speakers:
        others = [s for s in speakers if s != att]
        for _ in range(trials_per_speaker):
            trials.append(
                {
                    "trial_id": f"nju_{tid}",
                    "attended": att,
                    "unattended": list(others),
                }
            )
            tid += 1
    return trials


def dataset_condition_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """BI and role counts for a constructed trial list."""
    att, unatt = counts_from_trials(trials)
    return {
        "n_trials": len(trials),
        "balance_index": balance_index(att, unatt),
        "attended_counts": dict(att),
        "unattended_counts": dict(unatt),
    }
