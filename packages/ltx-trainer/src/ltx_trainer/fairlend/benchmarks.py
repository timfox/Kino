"""Paper table anchors (Rathod et al., arXiv:2606.12435)."""

from __future__ import annotations

PAPER_ANCHORS = {
    "dataset_rows": 103_481,
    "overall_denial_rate": 0.239,
    "standard_income_race_bias": 0.0963,
    "fair_income_race_bias": 0.08,
    "fair_income_epsilon": 0.08,
    "fair_income_pof": 0.2943,
    "fair_loan_epsilon": 0.06,
    "fair_loan_pof": 0.2354,
    "top_denial_rule_confidence": 0.672,
    "top_denial_rule_lift": 2.81,
    "dir_flagged_pairs": 10,
    "dir_audited_pairs": 45,
    "cluster3_black_white_dir": 0.693,
}

TABLE_1_CLEANING = [
    {"step": 0, "operation": "Filter Chicago MSA 16984", "rows_after": 204_100},
    {"step": 1, "operation": "Originated (1) + denied (3) only", "rows_after": 134_894},
    {"step": 2, "operation": "Remove unavailable race/sex/ethnicity", "rows_after": 106_754},
    {"step": 3, "operation": "Cast financial fields; drop null income", "rows_after": 104_477},
    {"step": 4, "operation": "Clip income 0.5–99.5 percentile", "rows_after": 103_481},
]

TABLE_2_INCOME_QUARTILES = [
    {"quartile": "Q1 Low", "count": 26_305, "avg_income_k": 53, "denial_rate": 0.353},
    {"quartile": "Q2 Mid-Low", "count": 25_722, "avg_income_k": 87, "denial_rate": 0.240},
    {"quartile": "Q3 Mid-High", "count": 25_627, "avg_income_k": 129, "denial_rate": 0.210},
    {"quartile": "Q4 High", "count": 25_827, "avg_income_k": 278, "denial_rate": 0.151},
]

TABLE_2_DENIAL_BY_RACE = [
    {"race": "White", "count": 73_799, "pct": 0.713, "denial_rate": 0.209},
    {"race": "Black or African American", "count": 16_171, "pct": 0.156, "denial_rate": 0.386},
    {"race": "Asian", "count": 10_149, "pct": 0.098, "denial_rate": 0.216},
    {"race": "Joint", "count": 2_219, "pct": 0.021, "denial_rate": 0.151},
    {"race": "American Indian or Alaska Native", "count": 691, "pct": 0.007, "denial_rate": 0.408},
    {"race": "2 or more minority races", "count": 281, "pct": 0.003, "denial_rate": 0.491},
    {"race": "Native Hawaiian or Other Pacific Islander", "count": 171, "pct": 0.002, "denial_rate": 0.456},
]

TABLE_3_BINNING = [
    {
        "attribute": "Income",
        "method": "Standard",
        "race_bias": 0.0963,
        "sex_bias": 0.2514,
        "epsilon": None,
        "pof": 0.0148,
    },
    {
        "attribute": "Income",
        "method": "ε-biased",
        "race_bias": 0.08,
        "sex_bias": 0.2108,
        "epsilon": 0.08,
        "pof": 0.2943,
    },
    {
        "attribute": "Loan Amount",
        "method": "Standard",
        "race_bias": 0.0765,
        "sex_bias": None,
        "epsilon": None,
        "pof": 0.0428,
    },
    {
        "attribute": "Loan Amount",
        "method": "ε-biased",
        "race_bias": 0.06,
        "sex_bias": None,
        "epsilon": 0.06,
        "pof": 0.2354,
    },
]

TABLE_4_FPGROWTH = [
    {"metric": "Frequent itemsets", "standard": 819, "fair": 854},
    {"metric": "Association rules (conf≥0.50, lift≥1.0)", "standard": 2214, "fair": 2454},
    {"metric": "Denial-consequent rules", "standard": 3, "fair": 3},
    {"metric": "Rules with demographic antecedents", "standard": 0, "fair": 0},
    {"metric": "Average confidence", "standard": 0.754, "fair": 0.738},
    {"metric": "Average lift", "standard": 1.140, "fair": 1.208},
]

TABLE_5_TOP_DENIAL_RULES = [
    {
        "rule": "{DTI_High} ⇒ {denied}",
        "support": 0.107,
        "confidence": 0.672,
        "lift": 2.81,
    },
    {
        "rule": "{DTI_High, PrimaryRes} ⇒ {denied}",
        "support": 0.102,
        "confidence": 0.663,
        "lift": 2.78,
    },
    {
        "rule": "{DTI_High} ⇒ {PrimaryRes, denied}",
        "support": 0.102,
        "confidence": 0.637,
        "lift": 2.82,
    },
]

TABLE_6_CLUSTER_METRICS = [
    {"k": 2, "wcss": 300_965, "silhouette": 0.5904},
    {"k": 3, "wcss": 235_336, "silhouette": 0.4749},
    {"k": 4, "wcss": 200_088, "silhouette": 0.3546},
    {"k": 5, "wcss": 175_597, "silhouette": 0.4176},
    {"k": 6, "wcss": 155_723, "silhouette": 0.4085},
    {"k": 7, "wcss": 140_604, "silhouette": 0.3971},
    {"k": 8, "wcss": 130_595, "silhouette": 0.3796},
]

TABLE_7_CLUSTER_PROFILES = [
    {"cluster": 0, "size": 48_783, "avg_income_k": 95, "avg_loan_k": 206, "avg_dti": 41.7, "avg_cltv": 88.8, "denial_pct": 17.0},
    {"cluster": 1, "size": 20_223, "avg_income_k": 225, "avg_loan_k": 373, "avg_dti": 31.4, "avg_cltv": 80.5, "denial_pct": 10.0},
    {"cluster": 2, "size": 11_314, "avg_income_k": 80, "avg_loan_k": 150, "avg_dti": 63.6, "avg_cltv": 70.0, "denial_pct": 79.0},
    {"cluster": 3, "size": 15_680, "avg_income_k": 104, "avg_loan_k": 116, "avg_dti": 34.5, "avg_cltv": 42.7, "denial_pct": 24.0},
    {"cluster": 4, "size": 3_524, "avg_income_k": 519, "avg_loan_k": 843, "avg_dti": 30.6, "avg_cltv": 75.4, "denial_pct": 11.0},
]

TABLE_8_DIR_FINDINGS = [
    {"cluster": 2, "group": "Native Hawaiian / PI", "grp_denial": 0.9412, "ref_denial": 0.7681, "dir": 0.254, "size": 34, "flag": "DISPARATE_IMPACT"},
    {"cluster": 3, "group": "2+ minority races", "grp_denial": 0.5152, "ref_denial": 0.2032, "dir": 0.608, "size": 33, "flag": "DISPARATE_IMPACT"},
    {"cluster": 3, "group": "Black or African American", "grp_denial": 0.4478, "ref_denial": 0.2032, "dir": 0.693, "size": 2146, "flag": "DISPARATE_IMPACT"},
    {"cluster": 2, "group": "Black or African American", "grp_denial": 0.8491, "ref_denial": 0.7681, "dir": 0.651, "size": 2372, "flag": "DISPARATE_IMPACT"},
]

TABLE_9_DIR_SUMMARY = [
    {"attribute": "Race", "pairs_audited": 25, "mean_dir": 0.814, "min_dir": 0.254, "max_dir": 1.059},
    {"attribute": "Ethnicity", "pairs_audited": 10, "mean_dir": 0.932, "min_dir": 0.824, "max_dir": 1.006},
    {"attribute": "Sex", "pairs_audited": 10, "mean_dir": 1.094, "min_dir": 1.002, "max_dir": 1.411},
]


def benchmarks_bundle() -> dict[str, object]:
    return {
        "anchors": PAPER_ANCHORS,
        "table_1_cleaning": TABLE_1_CLEANING,
        "table_2_denial_by_race": TABLE_2_DENIAL_BY_RACE,
        "table_2_income_quartiles": TABLE_2_INCOME_QUARTILES,
        "table_3_binning": TABLE_3_BINNING,
        "table_4_fpgrowth": TABLE_4_FPGROWTH,
        "table_5_denial_rules": TABLE_5_TOP_DENIAL_RULES,
        "table_6_cluster_metrics": TABLE_6_CLUSTER_METRICS,
        "table_7_cluster_profiles": TABLE_7_CLUSTER_PROFILES,
        "table_8_dir_findings": TABLE_8_DIR_FINDINGS,
        "table_9_dir_summary": TABLE_9_DIR_SUMMARY,
    }
