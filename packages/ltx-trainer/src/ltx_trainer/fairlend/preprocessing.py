"""HMDA 2023 Chicago cleaning pipeline stub (Sec. 3, Table 1)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.fairlend.benchmarks import TABLE_1_CLEANING, TABLE_2_DENIAL_BY_RACE
from ltx_trainer.fairlend.config import FairLendConfig


@dataclass(frozen=True)
class CleaningSummary:
    steps: tuple[dict[str, object], ...]
    final_rows: int
    overall_denial_rate: float
    black_white_income_gap_pct: float


def cleaning_pipeline_summary(cfg: FairLendConfig | None = None) -> CleaningSummary:
    cfg = cfg or FairLendConfig()
    white = next(r for r in TABLE_2_DENIAL_BY_RACE if r["race"] == "White")
    black = next(r for r in TABLE_2_DENIAL_BY_RACE if r["race"] == "Black or African American")
    white_income_k = 139.0
    black_income_k = 99.0
    gap = (white_income_k - black_income_k) / white_income_k
    return CleaningSummary(
        steps=tuple(TABLE_1_CLEANING),
        final_rows=cfg.n_applications,
        overall_denial_rate=cfg.overall_denial_rate,
        black_white_income_gap_pct=gap,
    )


def map_dti_range(raw: str) -> float | None:
    """Map HMDA DTI range strings to midpoint numeric values."""
    if not raw or raw in {"NA", "Exempt", ""}:
        return None
    text = raw.strip().replace(" ", "")
    if text.endswith("%") and "-<" not in text and "-" not in text.replace("-<", ""):
        try:
            return float(text.rstrip("%"))
        except ValueError:
            return None
    if "-<" in text or ("-" in text and "%" in text):
        cleaned = text.replace("%", "").replace("<", "")
        parts = cleaned.split("-")
        nums: list[float] = []
        for p in parts:
            if not p:
                continue
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if len(nums) >= 2:
            return (nums[0] + nums[1]) / 2.0
        if len(nums) == 1:
            return nums[0]
    try:
        return float(text.replace("%", ""))
    except ValueError:
        return None
