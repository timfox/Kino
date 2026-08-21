"""Table 6 failure mode vocabularies per domain."""

from __future__ import annotations

FAILURE_MODES_ALFWORLD: tuple[str, ...] = (
    "repetitive_exploration",
    "wrong_target_location",
    "wrong_receptacle",
    "premature_give_up",
    "missing_precondition",
    "repeated_failed_action",
    "navigation_loop",
    "entity_confusion",
    "wrong_object_interaction",
    "exhaustive_exploration_failure",
    "action_format_error",
)

FAILURE_MODES_WEBSHOP: tuple[str, ...] = (
    "irrelevant_query",
    "wrong_product_selection",
    "wrong_attribute_selection",
    "missing_attribute_selection",
    "premature_purchase",
    "excessive_browsing",
    "repeated_query",
    "navigation_error",
    "price_constraint_violation",
    "action_format_error",
    "premature_termination",
)

FAILURE_MODES_SEARCH: tuple[str, ...] = (
    "wrong_answer",
    "insufficient_retrieval",
    "irrelevant_retrieval_query",
    "repeated_retrieval_query",
    "information_misinterpretation",
    "partial_answer",
    "hallucinated_answer",
    "premature_answer",
    "action_format_error",
)

FAILURE_MODES_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    "alfworld": FAILURE_MODES_ALFWORLD,
    "webshop": FAILURE_MODES_WEBSHOP,
    "search_qa": FAILURE_MODES_SEARCH,
    "search": FAILURE_MODES_SEARCH,
}

FAILURE_MODE_COUNTS: dict[str, int] = {k: len(v) for k, v in FAILURE_MODES_BY_DOMAIN.items()}
