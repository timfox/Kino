"""Capability-aligned instruction operators (Table 11)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.panoworld.config import CAPABILITY_FAMILIES, PANOSPACE_CATEGORIES


@dataclass
class InstructionSample:
    capability: str
    category: str
    question: str
    choices: tuple[str, ...]
    answer_idx: int
    bfov_target: Tensor | None = None  # [4] degrees for BFOV tasks


def _choice_labels(n: int) -> tuple[str, ...]:
    return tuple(chr(ord("A") + i) for i in range(n))


def sample_instruction_batch(
    batch_size: int,
    num_choices: int,
    *,
    device: torch.device | None = None,
) -> list[InstructionSample]:
    """Synthetic MC / BFOV samples covering capability families."""
    samples: list[InstructionSample] = []
    caps = list(CAPABILITY_FAMILIES)
    cats = list(PANOSPACE_CATEGORIES)
    for i in range(batch_size):
        cap = caps[i % len(caps)]
        cat = cats[i % len(cats)]
        choices = _choice_labels(num_choices)
        answer = i % num_choices
        bfov = None
        if cat == "bfov_localization":
            bfov = torch.tensor([10.0, 5.0, 20.0, 15.0], device=device)
        samples.append(
            InstructionSample(
                capability=cap,
                category=cat,
                question=f"[{cat}] Where is the target relative to the observer?",
                choices=choices,
                answer_idx=answer,
                bfov_target=bfov,
            )
        )
    return samples


def batch_to_tensors(
    samples: list[InstructionSample],
    num_choices: int,
    *,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    b = len(samples)
    answers = torch.tensor([s.answer_idx for s in samples], device=device, dtype=torch.long)
    cap_ids = torch.tensor(
        [CAPABILITY_FAMILIES.index(s.capability) for s in samples],
        device=device,
        dtype=torch.long,
    )
    bfov = torch.zeros(b, 4, device=device)
    bfov_mask = torch.zeros(b, dtype=torch.bool, device=device)
    for i, s in enumerate(samples):
        if s.bfov_target is not None:
            bfov[i] = s.bfov_target
            bfov_mask[i] = True
    return {
        "answers": answers,
        "capability_ids": cap_ids,
        "num_choices": torch.tensor(num_choices, device=device),
        "bfov_gt": bfov,
        "bfov_mask": bfov_mask,
    }
