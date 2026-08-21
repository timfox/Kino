"""ε-biased binning feasibility sweep (Sec. 5.1, Sec. 7 future work)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.fairlend.binning import fair_binning_dc, synthetic_chicago_income_sample
from ltx_trainer.fairlend.config import FairLendConfig


# Paper-reported feasibility for seven race groups on income (Sec. 5.1).
PAPER_EPSILON_SWEEP_SEVEN = [
    {"epsilon": 0.03, "feasible": False, "note": "D&C infeasible with 7 race groups"},
    {"epsilon": 0.04, "feasible": False, "note": "D&C infeasible"},
    {"epsilon": 0.05, "feasible": False, "note": "D&C infeasible"},
    {"epsilon": 0.08, "feasible": True, "race_bias": 0.08, "pof": 0.2943},
]

PAPER_BINARY_BLACK_EPSILON = 0.03


@dataclass(frozen=True)
class EpsilonSweepRow:
    epsilon: float
    feasible: bool
    race_bias: float | None
    price_of_fairness: float | None
    grouping: str


def collapse_black_nonblack(groups: list[int]) -> list[int]:
    """Binary race: 1 = Black (code 1), 0 = non-Black (paper future-work setting)."""
    return [1 if g == 1 else 0 for g in groups]


def sweep_epsilon_live(
    *,
    k: int = 5,
    epsilons: tuple[float, ...] = (0.03, 0.04, 0.05, 0.06, 0.08),
    binary: bool = False,
    sample_n: int = 5000,
) -> list[EpsilonSweepRow]:
    incomes, groups = synthetic_chicago_income_sample(n=sample_n)
    if binary:
        groups = collapse_black_nonblack(groups)
        max_groups = 2
    else:
        max_groups = 7
    rows: list[EpsilonSweepRow] = []
    for eps in epsilons:
        result = fair_binning_dc(incomes, groups, k=k, epsilon=eps, max_groups=max_groups)
        rows.append(
            EpsilonSweepRow(
                epsilon=eps,
                feasible=result is not None,
                race_bias=result.race_bias if result else None,
                price_of_fairness=result.price_of_fairness if result else None,
                grouping="binary_black" if binary else "seven_race",
            )
        )
    return rows


def sweep_summary_paper_aligned(cfg: FairLendConfig | None = None) -> dict[str, object]:
    """Combine live binary sweep with paper anchors for seven-group infeasibility."""
    cfg = cfg or FairLendConfig()
    live_binary = sweep_epsilon_live(epsilons=(0.03, 0.05, 0.08), binary=True, sample_n=3000)
    binary_at_003 = next(r for r in live_binary if r.epsilon == 0.03)
    return {
        "seven_group_sweep": PAPER_EPSILON_SWEEP_SEVEN,
        "seven_group_feasible_at_008_only": all(
            not r["feasible"] for r in PAPER_EPSILON_SWEEP_SEVEN if r["epsilon"] < 0.08
        ),
        "binary_black_epsilon": PAPER_BINARY_BLACK_EPSILON,
        "binary_feasible_at_003_live": binary_at_003.feasible,
        "binary_sweep": [r.__dict__ for r in live_binary],
        "k_bins": cfg.n_bins,
    }
