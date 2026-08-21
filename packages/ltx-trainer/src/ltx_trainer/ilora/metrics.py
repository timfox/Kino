"""Paper result tables — Sec. 6."""

from __future__ import annotations


def table1_molweni() -> list[dict[str, str | float]]:
    return [
        {"method": "Zero-shot", "F1": 56.32, "EM": 35.70},
        {"method": "MLE", "F1": 72.83, "EM": 57.78},
        {"method": "MAP", "F1": 72.66, "EM": 57.02},
        {"method": "BLOB", "F1": 70.80, "EM": 55.90},
        {"method": "MCD", "F1": 72.33, "EM": 57.51},
        {"method": "ENS", "F1": 72.38, "EM": 57.09},
        {"method": "iLoRA", "F1": 74.51, "EM": 60.57},
    ]


def table2_graph_error() -> list[dict[str, str | float]]:
    return [
        {"benchmark": "Molweni", "graph": "Random", "error_rate": 50.0},
        {"benchmark": "Molweni", "graph": "iLoRA", "error_rate": 26.7},
        {"benchmark": "IBD", "graph": "Random", "error_rate": 50.0},
        {"benchmark": "IBD", "graph": "iLoRA", "error_rate": 27.3},
    ]


def table3_ibd_diagnosis() -> list[dict[str, str | float]]:
    return [
        {"method": "MLE", "ECE": 0.2533, "F1_UC": 0.6071, "AUROC": 0.7617, "AUPRC": 0.7570},
        {"method": "MAP", "ECE": 0.2082, "F1_UC": 0.6496, "AUROC": 0.7637, "AUPRC": 0.7117},
        {"method": "MCD", "ECE": 0.2762, "F1_UC": 0.6341, "AUROC": 0.7428, "AUPRC": 0.7117},
        {"method": "ENS", "ECE": 0.1598, "F1_UC": 0.5794, "AUROC": 0.7574, "AUPRC": 0.7565},
        {"method": "BLOB", "ECE": 0.1570, "F1_UC": 0.5882, "AUROC": 0.7812, "AUPRC": 0.7577},
        {"method": "LAP", "ECE": 0.2031, "F1_UC": 0.6496, "AUROC": 0.7641, "AUPRC": 0.7122},
        {"method": "iLoRA", "ECE": 0.0980, "F1_UC": 0.6557, "AUROC": 0.7990, "AUPRC": 0.7617},
    ]


def table4_ablation() -> list[dict[str, str | float]]:
    return [
        {"variant": "MLE (vanilla LoRA)", "ECE": 0.2533, "F1_UC": 0.6071, "AUROC": 0.7617, "AUPRC": 0.7570},
        {"variant": "iLoRA (w/o Laplace)", "ECE": 0.1032, "F1_UC": 0.6341, "AUROC": 0.7557, "AUPRC": 0.7440},
        {"variant": "iLoRA (full)", "ECE": 0.0980, "F1_UC": 0.6557, "AUROC": 0.7990, "AUPRC": 0.7617},
    ]


def table5_tabular_baselines() -> list[dict[str, str | float]]:
    return [
        {"model": "Random Forest", "F1_UC": 0.5753, "AUROC": 0.6151, "AUPRC": 0.6467},
        {"model": "XGBoost", "F1_UC": 0.5292, "AUROC": 0.5823, "AUPRC": 0.6467},
        {"model": "MLP", "F1_UC": 0.4906, "AUROC": 0.5346, "AUPRC": 0.6214},
        {"model": "iLoRA", "F1_UC": 0.6557, "AUROC": 0.7990, "AUPRC": 0.7617},
    ]


def table8_inference_cost() -> list[dict[str, str | float]]:
    return [
        {"model": "LoRA (MLE)", "latency_ms": 377.2, "gpu_mb": 21263.9},
        {"model": "ENS", "latency_ms": 1144.6, "gpu_mb": 21263.9},
        {"model": "iLoRA", "latency_ms": 567.5, "gpu_mb": 21330.3},
    ]
