"""Resumable rubric vote cache for large MMAE omni eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ltx_trainer.mmae.config import MMAEConfig
from ltx_trainer.mmae.judger_backend import RubricJudger
from ltx_trainer.mmae.rubrics import score_rubric_majority
from ltx_trainer.mmae.sample import MMAESample, Rubric

VoteKey = tuple[str, int, int]


class JudgerVoteCache:
    """JSONL-backed cache of per-vote judger letters."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.votes_path = self.cache_dir / "votes.jsonl"
        self._votes: dict[VoteKey, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.votes_path.is_file():
            return
        for line in self.votes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["sample_id"]),
                int(row["rubric_index"]),
                int(row["vote_index"]),
            )
            self._votes[key] = str(row["choice"])

    def get_vote(self, sample_id: str, rubric_index: int, vote_index: int) -> str | None:
        return self._votes.get((sample_id, rubric_index, vote_index))

    def set_vote(
        self,
        sample_id: str,
        rubric_index: int,
        vote_index: int,
        choice: str,
        *,
        flush: bool = True,
    ) -> None:
        key = (sample_id, rubric_index, vote_index)
        self._votes[key] = choice
        if flush:
            with self.votes_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "rubric_index": rubric_index,
                            "vote_index": vote_index,
                            "choice": choice,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def stats(self) -> dict[str, int]:
        return {"cached_votes": len(self._votes)}


def import_rubric_votes(path: str | Path) -> JudgerVoteCache:
    """Load external vote JSONL into an in-memory cache (no directory required)."""
    cache = JudgerVoteCache(Path(path).parent / ".import_cache")
    cache.votes_path = Path(path)
    cache._votes.clear()
    cache._load()
    return cache


def merge_vote_caches(*caches: JudgerVoteCache) -> JudgerVoteCache:
    merged = JudgerVoteCache(Path.cwd() / ".mmae_vote_merge")
    for cache in caches:
        merged._votes.update(cache._votes)
    return merged


def score_rubric_cached(
    sample_id: str,
    rubric_index: int,
    rubric: Rubric,
    *,
    judger: RubricJudger | None,
    cache: JudgerVoteCache | None,
    cfg: MMAEConfig,
    seed: int,
    resume: bool = True,
) -> dict[str, Any]:
    p = cfg.params
    if cache is None or judger is None:
        return score_rubric_majority(
            rubric,
            judger=judger,
            votes=p.majority_votes,
            threshold=p.majority_threshold,
            shuffle=p.shuffle_choices,
            seed=seed,
        )

    import random

    from ltx_trainer.mmae.rubrics import _seed_int, shuffle_choices

    rng = random.Random(_seed_int(seed, hash(rubric.question) & 0xFFFF))
    choices = rubric.all_choices()
    right_letter = chr(65 + choices.index(rubric.right_choice))
    if p.shuffle_choices:
        choices, right_letter = shuffle_choices(rubric, rng)

    ballot: list[str] = []
    vote_rights: list[str] = []
    for vote_index in range(p.majority_votes):
        cached = cache.get_vote(sample_id, rubric_index, vote_index) if resume else None
        if cached is not None:
            ballot.append(cached)
            if p.shuffle_choices:
                vote_rng = random.Random(_seed_int(seed, hash(rubric.question) & 0xFFFF, vote_index))
                _, vote_right = shuffle_choices(rubric, vote_rng)
                vote_rights.append(vote_right)
            else:
                vote_rights.append(right_letter)
            continue
        vote_rng = random.Random(_seed_int(seed, hash(rubric.question) & 0xFFFF, vote_index))
        if p.shuffle_choices:
            vote_choices, vote_right = shuffle_choices(rubric, vote_rng)
        else:
            vote_choices, vote_right = choices, right_letter
        letter = judger.choose(rubric, vote_choices, seed=seed + vote_index)
        cache.set_vote(sample_id, rubric_index, vote_index, letter)
        ballot.append(letter)
        vote_rights.append(vote_right)

    matches = sum(1 for b, vr in zip(ballot, vote_rights, strict=True) if b == vr)
    score = 1 if matches >= p.majority_threshold else 0
    return {
        "category": rubric.category.value,
        "score": score,
        "votes": ballot,
        "majority": matches,
        "right_letter": right_letter,
        "cached": resume,
    }


def score_sample_cached(
    sample: MMAESample,
    *,
    judger: RubricJudger,
    cache: JudgerVoteCache | None = None,
    cfg: MMAEConfig | None = None,
    seed: int = 0,
    resume: bool = True,
) -> dict[str, Any]:
    cfg = cfg or MMAEConfig()
    rubric_scores: list[dict[str, Any]] = []
    for i, rubric in enumerate(sample.rubrics):
        rubric_scores.append(
            score_rubric_cached(
                sample.sample_id,
                i,
                rubric,
                judger=judger,
                cache=cache,
                cfg=cfg,
                seed=seed + i,
                resume=resume,
            )
        )
    return {"sample_id": sample.sample_id, "rubrics": rubric_scores}
