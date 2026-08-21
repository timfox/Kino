"""Paper tables and reference curves — Sec. 3–4."""

from __future__ import annotations


def fig3_gap_curves() -> list[dict[str, int | float]]:
    """Vision vs language gap across N — Fig. 3."""
    rows: list[dict[str, int | float]] = []
    for n in range(0, 121, 5):
        v_gap = 0.0
        l_gap = 0.0 if n <= 49 else min(10.0, max(2.0, (n - 49) * 0.35))
        rows.append({"n": n, "vision_gap": v_gap, "language_gap": l_gap})
    return rows


def fig4_comparative_counting() -> list[dict[str, int | float]]:
    """Enumeration vs comparison accuracy — Fig. 4."""
    rows: list[dict[str, int | float]] = []
    for n in range(0, 121, 10):
        enum_acc = 100.0 if n <= 49 else (100.0 if n == 99 else 0.0)
        cmp_acc = 100.0 if n <= 49 else (92.0 if n <= 99 else 55.0)
        rows.append({"n": n, "enumeration_accuracy": enum_acc, "comparison_accuracy": cmp_acc})
    return rows


def fig7_qwen_baseline() -> list[dict[str, int | float]]:
    """Foundation model visual/text counting — Fig. 7."""
    rows: list[dict[str, int | float]] = []
    for n in range(0, 21):
        text_acc = 100.0 if n <= 15 else 50.0
        if n <= 5:
            vision_acc = 100.0
        elif n <= 8:
            vision_acc = max(0.0, 100.0 - (n - 5) * 25)
        else:
            vision_acc = 5.0
        rows.append({"n": n, "text_accuracy": text_acc, "vision_accuracy": vision_acc})
    return rows


def fig8_qwen_gaps() -> list[dict[str, int | float]]:
    """Qwen3-VL vision/language gaps — Fig. 8."""
    rows: list[dict[str, int | float]] = []
    for n in range(0, 21):
        v_gap = 0.05 if n > 0 else 0.0
        l_gap = 0.0 if n <= 4 else min(8.0, (n - 4) * 1.2)
        rows.append({"n": n, "vision_gap": v_gap, "language_gap": l_gap})
    return rows


def fig2_baseline_paradox() -> list[dict[str, int | float | str]]:
    """Text perfect to 99; vision collapses after 49."""
    rows: list[dict[str, int | float | str]] = []
    for n in range(0, 121, 10):
        text_acc = 100.0 if n <= 99 else 0.0
        vision_acc = 100.0 if n <= 49 else (100.0 if n == 99 else 0.0)
        rows.append({"n": n, "text_accuracy": text_acc, "vision_accuracy": vision_acc})
    return rows


def fig5_attractor_distribution() -> list[dict[str, str | float]]:
    """Error topology attractors — Fig. 5."""
    return [
        {"token": "49", "frequency_pct": 27.79},
        {"token": "Others", "frequency_pct": 32.41},
        {"token": "99", "frequency_pct": 12.02},
        {"token": "90", "frequency_pct": 8.59},
        {"token": "9", "frequency_pct": 8.23e-2},
        {"token": "58", "frequency_pct": 1.84},
    ]


def fig6_circuit_overlap() -> dict[str, float]:
    return {"disjoint_critical_heads_pct": 95.7, "n_examples": 6400}


def fig9_layer_probe_synthetic() -> list[dict[str, int | float]]:
    return [
        {"layer": "vision_last", "probe_accuracy": 100.0},
        {"layer": "decoder_1", "probe_accuracy": 100.0},
        {"layer": "decoder_2", "probe_accuracy": 100.0},
    ]


def fig9_layer_probe_qwen() -> list[dict[str, int | float]]:
    rows = [{"layer": "visual_module", "probe_accuracy": 99.37}]
    for layer in range(1, 41):
        rows.append({"layer": f"llm_{layer}", "probe_accuracy": 99.0})
    for layer in range(41, 63):
        rows.append({"layer": f"llm_{layer}", "probe_accuracy": max(85.0, 99.0 - (layer - 40) * 0.5)})
    return rows


def steering_curve_d11() -> list[dict[str, int | float]]:
    """Fig. 11 intervention accuracy."""
    return [
        {"k": 1, "steering_accuracy_pct": 100.0},
        {"k": 2, "steering_accuracy_pct": 100.0},
        {"k": 3, "steering_accuracy_pct": 100.0},
        {"k": 4, "steering_accuracy_pct": 98.0},
        {"k": 5, "steering_accuracy_pct": 77.2},
    ]


def three_stages() -> list[dict[str, str]]:
    return [
        {"stage": 1, "name": "visual_individuation", "hypothesis": "A", "verdict": "rejected"},
        {"stage": 2, "name": "magnitude_awareness", "hypothesis": "B", "verdict": "rejected"},
        {"stage": 3, "name": "symbolic_mapping", "hypothesis": "C", "verdict": "confirmed"},
    ]
