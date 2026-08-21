"""Balance index (BI) for EEG-AAD datasets — Eq. (1)."""

from __future__ import annotations

from collections.abc import Mapping


def balance_index(
    attended_counts: Mapping[str, int],
    unattended_counts: Mapping[str, int],
) -> float:
    """Compute BI in [0, 1]; 0 = perfectly balanced, 1 = extreme imbalance."""
    audio_ids = set(attended_counts) | set(unattended_counts)
    if not audio_ids:
        return 0.0
    terms: list[float] = []
    for stimulus_id in audio_ids:
        n_att = int(attended_counts.get(stimulus_id, 0))
        n_unatt = int(unattended_counts.get(stimulus_id, 0))
        denom = n_att + n_unatt
        if denom <= 0:
            continue
        terms.append(abs(n_att - n_unatt) / denom)
    if not terms:
        return 0.0
    return sum(terms) / len(audio_ids)


def counts_from_trials(
    trials: list[dict[str, object]],
    *,
    attended_key: str = "attended",
    unattended_key: str = "unattended",
) -> tuple[dict[str, int], dict[str, int]]:
    """Aggregate attended/unattended role counts from trial records."""
    att: dict[str, int] = {}
    unatt: dict[str, int] = {}
    for trial in trials:
        a = str(trial[attended_key])
        att[a] = att.get(a, 0) + 1
        u_raw = trial[unattended_key]
        if isinstance(u_raw, (list, tuple)):
            for u in u_raw:
                s = str(u)
                unatt[s] = unatt.get(s, 0) + 1
        else:
            s = str(u_raw)
            unatt[s] = unatt.get(s, 0) + 1
    return att, unatt
