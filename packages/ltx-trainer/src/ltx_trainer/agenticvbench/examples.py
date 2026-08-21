"""Toy task instances for AgenticVBench verifier demos."""

from __future__ import annotations

from ltx_trainer.agenticvbench.scorers import (
    assembly_score,
    repair_linear_reward,
    repair_window_reward,
    repurpose_rubric_score,
    sequencing_score,
    strict_sequence_match,
    timeline_range_match,
)


def demo_assembly_perfect() -> dict:
    """All storyboard slots pick golden clip among k=3 candidates."""
    slots, k = 5, 3
    score = assembly_score(slots, slots, k)
    return {"correct_slots": slots, "total_slots": slots, "candidates_per_slot": k, "score": score}


def demo_sequencing_perfect() -> dict:
    """Nine-clip reorder matching paper case-study scale."""
    true_order = list(range(9))
    pred_order = list(range(9))
    shuffled = [3, 7, 1, 8, 0, 4, 2, 6, 5]
    recovered = list(range(9))
    return {
        "n_clips": 9,
        "shuffled_input": shuffled,
        "predicted_order": recovered,
        "true_order": true_order,
        "score": sequencing_score(recovered, true_order),
        "strict_match": strict_sequence_match(recovered, true_order),
        "score_from_shuffled": sequencing_score(shuffled, true_order),
    }


def demo_repair_window() -> dict:
    """Simulated audio defect fix inside known window."""
    broken, golden, fixed = 0.2, 0.95, 0.88
    clean_broken, clean_golden, clean_fixed = 0.98, 0.99, 0.97
    in_s = repair_linear_reward(fixed, broken, golden)
    out_s = repair_linear_reward(clean_fixed, clean_broken, clean_golden)
    reward = repair_window_reward(in_s, out_s)
    return {
        "in_window_score": in_s,
        "out_window_score": out_s,
        "reward": reward,
    }


def demo_repair_timeline() -> dict:
    """F1 broadcast swapped-segment ranges (Appendix B.2 style)."""
    gold = [(156.46, 176.18), (176.18, 193.78)]
    pred = [(156.5, 176.2), (176.2, 193.8)]
    return {
        "gold_ranges": gold,
        "predicted_ranges": pred,
        "range_match": timeline_range_match(pred, gold, tolerance_s=0.5),
    }


def demo_repurpose_rubric() -> dict:
    """~30 binary items, partial pass illustrative of leaderboard band."""
    yes, total = 11, 30
    return {
        "yes_items": yes,
        "total_items": total,
        "score": repurpose_rubric_score(yes, total),
        "approx_points": 13,
        "max_points": 36,
    }
