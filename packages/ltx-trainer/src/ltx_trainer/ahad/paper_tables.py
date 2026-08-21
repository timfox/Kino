"""Paper table anchors and Remark thresholds."""

from __future__ import annotations


def remark2_thresholds() -> dict[str, float]:
    """Remark 2 ArmCBA interpretation (Section IV-B2)."""
    return {
        "acceptable_pct": 10.5,
        "promising_pct": 15.8,
        "superior_pct": 20.0,
        "reliable_auc_ap": 0.80,
        "acceptable_auc_ap_hi": 0.90,
    }


def table_ii_average() -> dict[str, float]:
    """Table II average row (Section IV-B2)."""
    return {
        "auc_up": 0.9166,
        "auc_ap": 0.7856,
        "armcba_pct": 15.8465,
    }


def table_ii_by_detector() -> dict[str, dict[str, float]]:
    """Selected Table II ArmCBA anchors (%)."""
    return {
        "LRASR": {"airport_i": 35.6747, "average": 24.3618},
        "SuperRPCA": {"airport_i": 28.4145, "muufl": 27.3609, "average": 25.8109},
        "LARTVAD": {"airport_i": 16.8352, "average": 15.9218},
        "BockNet": {"urban_ii": 4.7403, "average": 17.5860},
    }


def table_iii_ablation_average() -> dict[str, float]:
    """Table III average ArmCBA across datasets (Section IV-C)."""
    return {
        "case1_lipschitz_pag": 43.5439,
        "case2_sstv_lipschitz": 11.2291,
        "case3_sstv_pag": 0.3242,
        "case4_full": 16.3211,
        "case5_denoised": 26.3964,
    }


def dataset_catalog() -> list[dict[str, object]]:
    """Table I dataset metadata."""
    return [
        {"name": "Airport I", "location": "Los Angeles", "size": 100, "bands": 205, "anomalies": 87},
        {"name": "Airport II", "location": "Los Angeles", "size": 100, "bands": 205, "anomalies": 170},
        {"name": "MUUFL", "location": "Gulf Park Campus", "size": 150, "bands": 64, "anomalies": 259},
        {"name": "Urban I", "location": "Texas Coast", "size": 100, "bands": 204, "anomalies": 67},
        {"name": "Urban II", "location": "San Diego", "size": 100, "bands": 205, "anomalies": 272},
    ]
