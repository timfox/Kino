"""Paper benchmark tables (Sections 4–4.5, Table 1, Table 3)."""

from __future__ import annotations

from typing import Any

TABLE1_BIOML_DOMAIN = [
    {"domain": "Biomedical Imaging", "n": 4, "autoscientists": 45.75, "autoresearch": 39.60, "biomni": 19.04},
    {"domain": "Drug Discovery", "n": 9, "autoscientists": 64.52, "autoresearch": 46.16, "biomni": 47.91},
    {"domain": "Protein Engineering", "n": 6, "autoscientists": 96.97, "autoresearch": 96.97, "biomni": 93.94},
    {"domain": "Single Cell Omics", "n": 5, "autoscientists": 88.00, "autoresearch": 86.00, "biomni": 78.00},
]

TABLE3_ABLATIONS = [
    {"task": "TDC-hERG", "metric": "AUROC", "no_analyst": 0.738, "no_cross_agent": 0.819, "no_self_org": 0.807, "independent": 0.853, "full": 0.867},
    {"task": "Cell-Cell Communication", "metric": "Odds Ratio", "no_analyst": 0.858, "no_cross_agent": 0.908, "no_self_org": 0.628, "independent": 0.435, "full": 0.924},
    {"task": "Human Plasma-Protein Binding", "metric": "Pearson r", "no_analyst": 0.813, "no_cross_agent": 0.714, "no_self_org": 0.811, "independent": 0.784, "full": 0.873},
    {"task": "GPT Training", "metric": "val_bpb", "no_analyst": 0.9817, "no_cross_agent": 0.9814, "no_self_org": 0.9833, "independent": 0.9833, "full": 0.9777},
]

PROTEINGYM_TABLE2 = {
    "kermut_avg_spearman": 0.657,
    "autoscientists_kermut_avg_spearman": 0.700,
    "ace2_kermut": 0.747,
    "ace2_autoscientists": 0.840,
}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "bioml_table1_domains": TABLE1_BIOML_DOMAIN,
        "bioml_mean_leaderboard_pct": {
            "autoscientists": 74.40,
            "autoresearch": 66.07,
            "delta_pp": 8.33,
        },
        "gpt_nanochat": {
            "baseline_val_bpb": 0.998,
            "target_val_bpb": 0.978,
            "experiments_autoscientists": 34,
            "experiments_autoresearch": 65,
            "speedup_at_target": 1.9,
            "from_champion_keeps": {"autoscientists": 7, "autoresearch": 0},
            "from_champion_final_bpb": {"autoscientists": 0.9730, "autoresearch": 0.9777},
        },
        "proteingym": PROTEINGYM_TABLE2,
        "ablations_table3": TABLE3_ABLATIONS,
        "default_team": {"analysts": 3, "experiment_agents": 6},
    }
