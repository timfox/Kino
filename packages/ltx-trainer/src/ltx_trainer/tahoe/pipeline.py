"""Tahoe evaluation smoke and demo pipeline."""

from __future__ import annotations

from ltx_trainer.tahoe.attribution import format_strategy_credibility
from ltx_trainer.tahoe.atomic_diff import atomic_diff_schema_fields
from ltx_trainer.tahoe.baselines import (
    cross_model_transfer_summary,
    held_out_baseline_comparison,
    retrieve_top_k,
    example_bank_from_development,
)
from ltx_trainer.tahoe.benchmarks import (
    PAPER_ANCHORS,
    PARADIGM_COMPARISON,
    TABLE_2_MAIN,
    TABLE_3_HELD_OUT,
    TABLE_4_ATTRIBUTION,
    TABLE_5_BY_CATEGORY,
    benchmarks_bundle,
)
from ltx_trainer.tahoe.compiler import snowflake_syntax_valid, syntax_critic_loop
from ltx_trainer.tahoe.config import PAPER_ARXIV, PAPER_TITLE, TahoeConfig
from ltx_trainer.tahoe.deployment import run_deployment_ga4_demo
from ltx_trainer.tahoe.development import development_trace_records, run_development_batch
from ltx_trainer.tahoe.integration import framework_card, integration_bundle
from ltx_trainer.tahoe.learning import should_merge_deltas
from ltx_trainer.tahoe.management import batch_merge
from ltx_trainer.tahoe.metrics import metrics_from_table_row
from ltx_trainer.tahoe.query_pipeline import run_hint_guided_query
from ltx_trainer.tahoe.retrieval import filter_by_scope
from ltx_trainer.tahoe.running_example import (
    demo_ga4_visitors,
    demo_log10_transformation,
    demo_snowflake_quoting,
    demo_top_product_ties,
    seed_hint_bank,
)
from ltx_trainer.tahoe.strategy_attribution import attribution_summary, run_strategy_attribution


def evaluation_demo() -> dict[str, object]:
    bank = seed_hint_bank()
    cfg = TahoeConfig()
    ga4_scoped = filter_by_scope(bank.semantic, database_id="GA4")
    cred = format_strategy_credibility(bank.semantic[0].strategies[0])
    dev = run_development_batch()
    deploy = run_deployment_ga4_demo()
    held_out = held_out_baseline_comparison()
    bank_attr = run_strategy_attribution(bank)
    query = run_hint_guided_query("Count all orders placed", bank)
    gpt55_tahoe_metrics = metrics_from_table_row(
        next(r for r in TABLE_2_MAIN if r["backbone"] == "GPT-5.5" and r["config"] == "Tahoe")
    )
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "framework": framework_card(cfg),
        "integration": integration_bundle(),
        "benchmarks": benchmarks_bundle(),
        "anchors": PAPER_ANCHORS,
        "table_2": TABLE_2_MAIN,
        "table_3_held_out": TABLE_3_HELD_OUT,
        "table_4_attribution": TABLE_4_ATTRIBUTION,
        "table_5_categories": TABLE_5_BY_CATEGORY,
        "paradigm_table": PARADIGM_COMPARISON,
        "hint_bank_counts": {
            "syntax": len(bank.syntax),
            "semantic": len(bank.semantic),
        },
        "demo_snowflake_quoting": demo_snowflake_quoting(),
        "demo_log10": demo_log10_transformation(),
        "demo_ga4": demo_ga4_visitors(),
        "demo_top_ties": demo_top_product_ties(),
        "development_batch": dev.__dict__,
        "development_diff_records": development_trace_records(),
        "atomic_diff_schema": atomic_diff_schema_fields(),
        "deployment_ga4": deploy,
        "held_out_baselines": held_out,
        "cross_model_transfer": cross_model_transfer_summary(),
        "attribution_summary": attribution_summary(bank_attr)[:5],
        "hint_guided_query": query.__dict__,
        "table2_gpt55_metrics": gpt55_tahoe_metrics.__dict__,
        "sqlgenie_retrieval_demo": [
            e.question for e in retrieve_top_k("count orders", example_bank_from_development())
        ],
        "ga4_scoped_hints": len(ga4_scoped),
        "strategy_credibility_sample": cred.__dict__,
        "merge_policy_success": should_merge_deltas(
            all_samples_correct=True,
            correct_under_temp=4,
            correct_under_frozen=2,
            max_iterations_reached=False,
        ),
        "batch_merge_ok": len(batch_merge(bank, []).semantic) == len(bank.semantic),
    }


def evaluation_smoke() -> dict[str, bool]:
    bank = seed_hint_bank()
    cfg = TahoeConfig()
    snow = demo_snowflake_quoting()
    log10 = demo_log10_transformation()
    ga4 = demo_ga4_visitors()
    ties = demo_top_product_ties()
    dev = run_development_batch()
    deploy = run_deployment_ga4_demo()
    held_out = held_out_baseline_comparison()
    transfer = cross_model_transfer_summary()
    query = run_hint_guided_query("Count all orders placed", bank)
    bad_sql = "SELECT COUNT(*) FROM SALES.ORDERS"
    _, critic_rounds, fixed_ok = syntax_critic_loop(bad_sql, bank)
    gpt55_tahoe = next(r for r in TABLE_2_MAIN if r["backbone"] == "GPT-5.5" and r["config"] == "Tahoe")
    gpt55_vanilla = next(r for r in TABLE_2_MAIN if r["backbone"] == "GPT-5.5" and r["config"] == "Vanilla")
    attr_on = TABLE_4_ATTRIBUTION[1]["pass_rate"]
    attr_off = TABLE_4_ATTRIBUTION[0]["pass_rate"]
    checks = {
        "hint_bank_sizes": len(bank.syntax) == cfg.syntax_hint_count and len(bank.semantic) == cfg.semantic_hint_count,
        "snowflake_quoting_demo": snow["quoted_identifiers"] is True,
        "log10_add_one_demo": log10["uses_add_one"] is True,
        "ga4_pseudo_id_demo": ga4["uses_pseudo_id"] is True,
        "top_product_ties_demo": ties["avoids_limit_one"] and ties["uses_max_join"],
        "development_batch_grows_bank": dev.final_syntax_count >= 1,
        "deployment_semantic_learn": deploy["action"] == "semantic_learn",
        "held_out_tahoe_beats_rag": held_out["tahoe_vs_vanilla_pass_rate_pp"]
        > held_out["rag_vs_vanilla_pass_rate_pp"],
        "cross_model_doubao_lift": any(t["backbone"] == "Doubao-2.0-lite" and t["pass_rate_delta_pp"] > 15 for t in transfer),
        "table2_tahoe_beats_vanilla": gpt55_tahoe["pass_rate"] > gpt55_vanilla["pass_rate"],
        "table2_syntax_perfect": gpt55_tahoe["syntax_pass"] == 1.0,
        "table4_attribution_lift": attr_on > attr_off,
        "held_out_syntax_gain": TABLE_3_HELD_OUT[2]["syntax_pass"] > TABLE_3_HELD_OUT[0]["syntax_pass"],
        "framework_card_ok": framework_card()["paper"]["arxiv"] == PAPER_ARXIV,
        "scope_filter_ga4": len(filter_by_scope(bank.semantic, database_id="GA4")) >= 1,
        "atomic_diff_schema_ok": len(atomic_diff_schema_fields()) == 9,
        "syntax_critic_repair": fixed_ok and critic_rounds >= 1,
        "hint_query_valid_sql": query.syntax_valid and snowflake_syntax_valid(query.final_sql),
        "attribution_pass_updates_stats": any(
            s.eval_stats and s.eval_stats.helped
            for h in run_strategy_attribution(bank).semantic
            for s in h.strategies
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks
