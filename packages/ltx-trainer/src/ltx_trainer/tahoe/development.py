"""Development-phase batch learning stub (Sec. 5.2, Fig. 3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.tahoe.atomic_diff import atomic_diff_to_record, make_limit_rank_diff
from ltx_trainer.tahoe.hints import HintBank, SemanticHint, SyntaxHint
from ltx_trainer.tahoe.learning import (
    AtomicDiff,
    TemporaryHintBank,
    diff_from_syntax_error,
    should_merge_deltas,
)
from ltx_trainer.tahoe.management import batch_merge_hints
from ltx_trainer.tahoe.running_example import paper_semantic_hints, paper_syntax_hints


@dataclass
class LearningExample:
    question: str
    ground_truth_sql: str
    database_id: str | None = None


@dataclass
class PerExampleLearningTrace:
    question: str
    iterations: int
    syntax_diffs: list[AtomicDiff] = field(default_factory=list)
    semantic_diffs: list[AtomicDiff] = field(default_factory=list)
    merge_eligible: bool = False
    correct_under_temp: int = 0
    correct_under_frozen: int = 0


@dataclass
class DevelopmentBatchResult:
    batches: int
    examples_processed: int
    merge_eligible: int
    final_syntax_count: int
    final_semantic_count: int


def _init_temp_bank(frozen: HintBank) -> TemporaryHintBank:
    return TemporaryHintBank(
        syntax_additions=[h.hint_id for h in frozen.syntax],
        semantic_additions=[h.hint_id for h in frozen.semantic],
        revisions=[],
    )


def learn_single_example(
    example: LearningExample,
    frozen: HintBank,
    *,
    samples: int = 4,
    max_iterations: int = 3,
) -> tuple[PerExampleLearningTrace, tuple[SyntaxHint, ...], tuple[SemanticHint, ...]]:
    """Simulate Fig. 3 loop for one supervised example."""
    trace = PerExampleLearningTrace(question=example.question, iterations=0)
    temp = _init_temp_bank(frozen)
    syntax_deltas: tuple[SyntaxHint, ...] = ()
    semantic_deltas: tuple[SemanticHint, ...] = ()

    wrong_sql = "SELECT name, sales FROM PRODUCTS ORDER BY sales DESC LIMIT 1"
    if "count" in example.question.lower() and "order" in example.question.lower():
        wrong_sql = "SELECT COUNT(*) AS total FROM SALES.ORDERS"
        trace.syntax_diffs.append(
            diff_from_syntax_error(
                "Invalid identifier 'ORDERS'",
                wrong_sql,
                example.ground_truth_sql,
            )
        )
        temp.syntax_additions.append("syn_quote_identifiers")
        trace.correct_under_temp = samples
        trace.correct_under_frozen = 0
        trace.merge_eligible = True
        trace.iterations = 1
        syntax_deltas = paper_syntax_hints()
        return trace, syntax_deltas, semantic_deltas

    if "top" in example.question.lower():
        trace.semantic_diffs.append(make_limit_rank_diff())
        temp.semantic_additions.append("sem_top_product_ties")
        trace.correct_under_temp = samples
        trace.correct_under_frozen = 1
        trace.iterations = 2
    else:
        trace.correct_under_temp = 2
        trace.correct_under_frozen = 2

    trace.merge_eligible = should_merge_deltas(
        all_samples_correct=trace.correct_under_temp == samples,
        correct_under_temp=trace.correct_under_temp,
        correct_under_frozen=trace.correct_under_frozen,
        max_iterations_reached=trace.iterations >= max_iterations,
    )
    if trace.merge_eligible and trace.semantic_diffs:
        semantic_deltas = paper_semantic_hints()[2:3]
    return trace, syntax_deltas, semantic_deltas


def run_development_batch(
    examples: list[LearningExample] | None = None,
    *,
    batch_count: int = 2,
) -> DevelopmentBatchResult:
    """Batch-sequential protocol (Sec. 5.5): H(t) → H(t+1) after each batch."""
    if examples is None:
        examples = [
            LearningExample(
                "Count all orders placed",
                'SELECT COUNT(*) AS "total"\nFROM "SALES"."ORDERS";',
            ),
            LearningExample(
                "Show me the top-selling product including ties",
                (
                    'WITH "top_sales" AS (SELECT MAX("sales") AS "max_sales" FROM "PRODUCTS")\n'
                    'SELECT "name", "sales" FROM "PRODUCTS" p JOIN "top_sales" t ON p."sales" = t."max_sales";'
                ),
            ),
        ]
    bank = HintBank(syntax=(), semantic=())
    merge_eligible = 0
    for _ in range(batch_count):
        batch_syntax: tuple[SyntaxHint, ...] = ()
        batch_semantic: tuple[SemanticHint, ...] = ()
        for ex in examples:
            trace, syntax_deltas, semantic_deltas = learn_single_example(ex, bank)
            if trace.merge_eligible:
                merge_eligible += 1
                batch_syntax += syntax_deltas
                batch_semantic += semantic_deltas
        if batch_syntax or batch_semantic:
            bank = batch_merge_hints(
                bank,
                syntax_deltas=batch_syntax,
                semantic_deltas=batch_semantic,
            )
    return DevelopmentBatchResult(
        batches=batch_count,
        examples_processed=len(examples) * batch_count,
        merge_eligible=merge_eligible,
        final_syntax_count=len(bank.syntax),
        final_semantic_count=len(bank.semantic),
    )


def development_trace_records() -> list[dict[str, str]]:
    ex = LearningExample(
        "Count all orders placed",
        'SELECT COUNT(*) AS "total"\nFROM "SALES"."ORDERS";',
    )
    trace, _, _ = learn_single_example(ex, HintBank(syntax=(), semantic=()))
    return [atomic_diff_to_record(d) for d in trace.syntax_diffs]
