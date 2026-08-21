"""Subject-level majority vote — §3.1."""

from __future__ import annotations

from collections import Counter


def subject_label_from_segments(
    segment_labels: list[int],
    segment_probs: list[float],
) -> tuple[int, float]:
    """Majority vote; tie-break by mean probability of winning label."""
    if not segment_labels:
        return 0, 0.0
    counts = Counter(segment_labels)
    top = counts.most_common()
    winners = [lab for lab, c in top if c == top[0][1]]
    if len(winners) == 1:
        label = winners[0]
    else:
        means = {
            lab: sum(p for l, p in zip(segment_labels, segment_probs, strict=True) if l == lab)
            / max(sum(1 for l in segment_labels if l == lab), 1)
            for lab in winners
        }
        label = max(means, key=means.get)
    prob = sum(
        p for l, p in zip(segment_labels, segment_probs, strict=True) if l == label
    ) / max(sum(1 for l in segment_labels if l == label), 1)
    return label, float(prob)
