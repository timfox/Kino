"""ReceiptBench field schema and sub-task taxonomy (Sec. 3.3, Table 5)."""

from __future__ import annotations

from typing import Literal

FieldMetric = Literal["exact", "numeric", "semantic", "list"]
FieldTask = Literal["perception", "normalization", "reasoning", "structure"]

# field_name -> (sub_task, metric_type)
FIELD_SPEC: dict[str, tuple[FieldTask, FieldMetric]] = {
    "orig_start_time": ("perception", "semantic"),
    "orig_end_time": ("perception", "semantic"),
    "orig_invoice_time": ("perception", "semantic"),
    "orig_total": ("perception", "numeric"),
    "orig_curr": ("perception", "list"),
    "invoice_number": ("perception", "exact"),
    "tax_number": ("perception", "exact"),
    "seller_name": ("perception", "semantic"),
    "std_start_time": ("normalization", "exact"),
    "std_end_time": ("normalization", "exact"),
    "std_invoice_time": ("normalization", "exact"),
    "std_total": ("normalization", "numeric"),
    "type": ("reasoning", "exact"),
    "place": ("reasoning", "semantic"),
    "departure": ("reasoning", "semantic"),
    "arrival": ("reasoning", "semantic"),
    "std_curr": ("reasoning", "exact"),
    "seller_address": ("reasoning", "semantic"),
    "detail": ("structure", "list"),
}

PERCEPTION_FIELDS: tuple[str, ...] = tuple(
    k for k, (t, _) in FIELD_SPEC.items() if t == "perception"
)
NORMALIZATION_FIELDS: tuple[str, ...] = tuple(
    k for k, (t, _) in FIELD_SPEC.items() if t == "normalization"
)
REASONING_FIELDS: tuple[str, ...] = tuple(
    k for k, (t, _) in FIELD_SPEC.items() if t == "reasoning"
)
STRUCTURE_FIELDS: tuple[str, ...] = tuple(
    k for k, (t, _) in FIELD_SPEC.items() if t == "structure"
)

ALL_FIELDS: tuple[str, ...] = tuple(FIELD_SPEC.keys())


def fields_for_subtask(subtask: FieldTask) -> tuple[str, ...]:
    return tuple(k for k, (t, _) in FIELD_SPEC.items() if t == subtask)
