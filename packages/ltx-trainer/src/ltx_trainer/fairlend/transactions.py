"""HMDA application → FP-Growth transaction encoding (Sec. 4.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


TRANSACTION_FIELDS = (
    "outcome",
    "race",
    "sex",
    "ethnicity",
    "age_bracket",
    "income_bin",
    "loan_amount_bin",
    "dti_bin",
    "loan_type",
    "loan_purpose",
    "occupancy_type",
)


@dataclass(frozen=True)
class ApplicationRecord:
    denied: bool
    race: str
    sex: str
    ethnicity: str
    age_bracket: str
    income_k: float
    loan_amount_k: float
    dti_numeric: float
    loan_type: str
    loan_purpose: str
    occupancy_type: str


def classify_dti_bin(dti: float) -> str:
    if dti >= 43.0:
        return "DTI_High"
    if dti >= 30.0:
        return "DTI_Medium"
    return "DTI_Low"


def classify_income_bin(income_k: float) -> str:
    if income_k < 75:
        return "Income_Q1"
    if income_k < 120:
        return "Income_Q2"
    if income_k < 200:
        return "Income_Q3"
    return "Income_Q4"


def classify_loan_bin(loan_k: float) -> str:
    if loan_k < 150:
        return "Loan_Small"
    if loan_k < 300:
        return "Loan_Medium"
    return "Loan_Large"


def build_transaction(record: ApplicationRecord) -> frozenset[str]:
    """Prefix feature names for unique FP-Growth items (Sec. 4.3)."""
    outcome = "denied" if record.denied else "originated"
    items = {
        f"outcome={outcome}",
        f"race={record.race}",
        f"sex={record.sex}",
        f"ethnicity={record.ethnicity}",
        f"age_bracket={record.age_bracket}",
        f"income_bin={classify_income_bin(record.income_k)}",
        f"loan_amount_bin={classify_loan_bin(record.loan_amount_k)}",
        f"dti_bin={classify_dti_bin(record.dti_numeric)}",
        f"loan_type={record.loan_type}",
        f"loan_purpose={record.loan_purpose}",
        f"occupancy_type={record.occupancy_type}",
    }
    return frozenset(items)


def synthetic_chicago_applications(n: int = 200, seed: int = 42) -> tuple[ApplicationRecord, ...]:
    import random

    rng = random.Random(seed)
    rows: list[ApplicationRecord] = []
    for _ in range(n):
        race = rng.choices(
            ["White", "Black or African American", "Asian"],
            weights=[0.713, 0.156, 0.098],
        )[0]
        income = rng.gauss(139 if race == "White" else 99, 35)
        dti = rng.gauss(45 if income < 90 else 32, 8)
        denied = dti >= 43 or (race == "Black or African American" and rng.random() < 0.15)
        rows.append(
            ApplicationRecord(
                denied=denied,
                race=race,
                sex=rng.choice(["Male", "Female"]),
                ethnicity=rng.choice(["Not Hispanic or Latino", "Hispanic or Latino"]),
                age_bracket=rng.choice(["25-34", "35-44", "45-54"]),
                income_k=max(16.0, income),
                loan_amount_k=max(50.0, rng.gauss(220, 80)),
                dti_numeric=max(0.0, dti),
                loan_type="Conventional",
                loan_purpose="Home purchase",
                occupancy_type="PrimaryRes",
            )
        )
    return tuple(rows)


def transactions_from_records(records: Sequence[ApplicationRecord]) -> list[frozenset[str]]:
    return [build_transaction(r) for r in records]
