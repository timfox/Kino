"""Paper result tables — Sec. 4–5."""

from __future__ import annotations


def table1_object_discovery() -> list[dict[str, str | float]]:
    """Table 1: MOVi-C/E + YTVIS-HQ discovery metrics."""
    rows: list[dict[str, str | float]] = []
    for method, movic, movie, ytvis in [
        ("VideoSAUR", (41.9, 17.4, 33.8), (53.3, 34.6, 49.2), (16.1, 8.3, 29.9)),
        ("SlotContrast", (64.6, 29.9, 37.2), (59.9, 70.6, 49.4), (27.7, 20.7, 33.0)),
        ("RandSF.Q", (65.4, 30.5, 40.1), (67.4, 82.1, 58.0), (29.2, 23.0, 37.6)),
        ("RandSF.Q+ECC", (53.8, 34.0, 40.0), (46.6, 45.1, 57.2), (20.5, 13.3, 37.0)),
        ("RandSF.Q+ICC", (73.2, 41.6, 40.6), (67.4, 80.5, 60.1), (32.9, 26.3, 39.2)),
        ("SmoothSA", (50.9, 36.7, 42.4), (69.0, 73.6, 63.0), (31.7, 28.6, 38.9)),
        ("SmoothSA+ECC", (44.1, 35.0, 40.2), (70.4, 65.2, 59.8), (30.6, 23.9, 38.0)),
        ("SmoothSA+ICC", (52.1, 35.4, 42.1), (71.2, 74.0, 60.2), (33.4, 28.7, 39.6)),
    ]:
        rows.append(
            {
                "method": method,
                "MOVi-C_ARI": movic[0],
                "MOVi-E_ARI": movie[0],
                "YTVIS_ARI": ytvis[0],
                "MOVi-C_ARIfg": movic[1],
                "MOVi-E_ARIfg": movie[1],
                "YTVIS_ARIfg": ytvis[1],
                "MOVi-C_mBO": movic[2],
                "MOVi-E_mBO": movie[2],
                "YTVIS_mBO": ytvis[2],
            }
        )
    return rows


def table2_efficiency() -> list[dict[str, str | float]]:
    return [
        {"variant": "RandSF.Q @ MOVi-E", "train_gb": 24.2, "eval_gb": 8.0, "train_min": 7.9, "eval_min": 1.2},
        {"variant": "+ECC", "train_gb": 24.6, "eval_gb": 8.1, "train_min": 8.0, "eval_min": 1.3},
        {"variant": "+ICC", "train_gb": 24.7, "eval_gb": 8.4, "train_min": 8.4, "eval_min": 1.3},
    ]


def table3_recognition() -> list[dict[str, str | float]]:
    return [
        {"method": "RandSF.Q+MLP", "top1": 90.5, "top3": 97.9, "iou": 50.6, "n_match": 8979},
        {"method": "RandSF.Q+ICC+MLP", "top1": 91.6, "top3": 97.7, "iou": 52.5, "n_match": 9233},
        {"method": "SmoothSA+MLP", "top1": 90.4, "top3": 97.6, "iou": 42.6, "n_match": 8957},
        {"method": "SmoothSA+ICC+MLP", "top1": 91.5, "top3": 97.9, "iou": 47.0, "n_match": 9112},
    ]


def table4_ablation_movc() -> list[dict[str, str | float]]:
    return [
        {"variant": "RandSF.Q", "ARI": 65.4, "ARIfg": 67.4, "mBO": 29.2, "mIoU": 26.8},
        {"variant": "+ ECC", "ARI": 53.8, "ARIfg": 46.6, "mBO": 20.5, "mIoU": 17.6},
        {"variant": "+ Hungarian ECC", "ARI": 59.4, "ARIfg": 47.2, "mBO": 22.4, "mIoU": 18.3},
        {"variant": "+ Non-Chain Recon.", "ARI": 68.3, "ARIfg": 65.1, "mBO": 29.1, "mIoU": 27.4},
        {"variant": "+ ICC", "ARI": 73.2, "ARIfg": 67.4, "mBO": 32.9, "mIoU": 30.3},
    ]


def table5_collapse_ytvis() -> list[dict[str, str | float]]:
    return [
        {"variant": "RandSF.Q", "slot_recon_x100": 93.4, "diversity": 29.8, "variance": 51.5},
        {"variant": "+ ECC", "slot_recon_x100": 90.4, "diversity": 19.5, "variance": 67.3, "note": "collapse"},
        {"variant": "+ ICC", "slot_recon_x100": 91.6, "diversity": 30.0, "variance": 53.7},
    ]
