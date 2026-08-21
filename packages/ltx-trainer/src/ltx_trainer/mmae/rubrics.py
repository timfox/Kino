"""Rubric scoring with majority vote and choice shuffling."""

from __future__ import annotations

import random
from typing import Any

from ltx_trainer.mmae.config import MMAEConfig
from ltx_trainer.mmae.judger import mock_judger_choice
from ltx_trainer.mmae.judger_backend import MockTableJudger, RubricJudger
from ltx_trainer.mmae.sample import MMAESample, Rubric


def shuffle_choices(
    rubric: Rubric,
    rng: random.Random,
) -> tuple[list[str], str]:
    choices = rubric.all_choices()
    rng.shuffle(choices)
    right_letter = chr(65 + choices.index(rubric.right_choice))
    return choices, right_letter


def _seed_int(*parts: int) -> int:
    value = 0
    for part in parts:
        value = (value * 1_000_003) ^ (part & 0xFFFFFFFF)
    return value & 0xFFFFFFFF


def score_rubric_majority(
    rubric: Rubric,
    *,
    p_correct: float | None = None,
    judger: RubricJudger | None = None,
    votes: int = 3,
    threshold: int = 2,
    shuffle: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    rng = random.Random(_seed_int(seed, hash(rubric.question) & 0xFFFF))
    choices = rubric.all_choices()
    right_letter = chr(65 + choices.index(rubric.right_choice))
    if shuffle:
        choices, right_letter = shuffle_choices(rubric, rng)
    ballot: list[str] = []
    vote_rights: list[str] = []
    for v in range(votes):
        vote_rng = random.Random(_seed_int(seed, hash(rubric.question) & 0xFFFF, v))
        if shuffle:
            vote_choices, vote_right = shuffle_choices(rubric, vote_rng)
        else:
            vote_choices, vote_right = choices, right_letter
        if judger is not None:
            letter = judger.choose(rubric, vote_choices, seed=seed + v)
        else:
            p_corr = p_correct if p_correct is not None else 0.5
            letter = mock_judger_choice(
                rubric.right_choice,
                vote_choices,
                p_correct=p_corr,
                rng=vote_rng,
            )
        ballot.append(letter)
        vote_rights.append(vote_right)
    matches = sum(1 for b, vr in zip(ballot, vote_rights, strict=True) if b == vr)
    score = 1 if matches >= threshold else 0
    return {
        "category": rubric.category.value,
        "score": score,
        "votes": ballot,
        "majority": matches,
        "right_letter": right_letter,
    }


def score_sample(
    sample: MMAESample,
    *,
    p_correct_if: float | None = None,
    p_correct_cr: float | None = None,
    judger: RubricJudger | None = None,
    cfg: MMAEConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = cfg or MMAEConfig()
    p = cfg.params
    if judger is None and (p_correct_if is None or p_correct_cr is None):
        raise ValueError("score_sample requires judger or both p_correct_if and p_correct_cr")
    table_judger: RubricJudger | None = judger
    if table_judger is None:
        table_judger = MockTableJudger(p_correct_if=p_correct_if or 0.0, p_correct_cr=p_correct_cr or 0.0)
    rubric_scores: list[dict[str, Any]] = []
    for i, rubric in enumerate(sample.rubrics):
        rubric_scores.append(
            score_rubric_majority(
                rubric,
                judger=table_judger,
                votes=p.majority_votes,
                threshold=p.majority_threshold,
                shuffle=p.shuffle_choices,
                seed=seed + i,
            )
        )
    return {"sample_id": sample.sample_id, "rubrics": rubric_scores}
