"""BES paper constants — arXiv:2605.28814, Embodied-Minds-Lab/BES."""

from __future__ import annotations

PAPER_ARXIV = "arXiv:2605.28814"
PAPER_TITLE = (
    "Self-Improving Language Models with Bidirectional Evolutionary Search"
)
GITHUB_REPO = "https://github.com/Embodied-Minds-Lab/BES"
PROJECT_PAGE = "https://guoweixu.com/bes/"

# Forward operator mixture (Section 3.1, shared across KK / MuSiQue)
OPERATOR_PROBS: dict[str, float] = {
    "expand": 0.70,
    "combine": 0.10,
    "delete": 0.05,
    "translocate": 0.075,
    "crossover": 0.075,
}

# Human / AlphaEvolve reference scores (Table 2, open problems)
CIRCLE_PACKING_SQUARE_HUMAN = 2.634
CIRCLE_PACKING_SQUARE_ALPHAEVOLVE = 2.635
CIRCLE_PACKING_RECT_HUMAN = 2.364
CIRCLE_PACKING_RECT_ALPHAEVOLVE = 2.3658
HEILBRONN_CONVEX_HUMAN = 0.0306
HEILBRONN_CONVEX_ALPHAEVOLVE = 0.0309

# ShinkaEvolve + BES on GPT-5 (Table 2 means / bests)
OPEN_PROBLEM_RESULTS: dict[str, dict[str, dict[str, float]]] = {
    "circle_packing_square": {
        "OpenEvolve": {"avg": 2.531, "best": 2.541},
        "GEPA": {"avg": 2.613, "best": 2.628},
        "ShinkaEvolve": {"avg": 2.464, "best": 2.541},
        "BES": {"avg": 2.623, "best": 2.632},
    },
    "circle_packing_rect": {
        "OpenEvolve": {"avg": 2.267, "best": 2.276},
        "GEPA": {"avg": 2.326, "best": 2.354},
        "ShinkaEvolve": {"avg": 2.335, "best": 2.358},
        "BES": {"avg": 2.349, "best": 2.360},
    },
    "heilbronn_convex": {
        "OpenEvolve": {"avg": 0.025, "best": 0.027},
        "GEPA": {"avg": 0.025, "best": 0.027},
        "ShinkaEvolve": {"avg": 0.023, "best": 0.026},
        "BES": {"avg": 0.026, "best": 0.027},
    },
}

# MuSiQue post-training (Table 1)
MUSIQUE_POST_TRAIN: dict[str, dict[str, float | str]] = {
    "Llama-3.2-3B-Instruct": {
        "base_acc": 4.0,
        "GRPO_acc": 2.1,
        "Tree-GRPO_acc": 3.9,
        "BES_acc": 7.0,
        "BES_valid_search": 2.31,
        "BES_finish_ratio": 0.97,
    },
    "Llama-3.1-8B-Instruct": {
        "base_acc": 6.6,
        "GRPO_acc": 5.6,
        "Tree-GRPO_acc": 7.4,
        "BES_acc": 10.4,
        "BES_valid_search": 2.11,
        "BES_finish_ratio": 0.94,
    },
}

# Cost analysis (Tables 3–4)
MUSIQUE_WALLTIME_SEC: dict[str, float] = {
    "GRPO": 64.0,
    "Tree-GRPO": 240.0,
    "BES": 309.0,
}
OPEN_PROBLEM_API_COST_USD: dict[str, dict[str, float]] = {
    "ShinkaEvolve": {
        "circle_packing_square": 13.0,
        "circle_packing_rect": 11.9,
        "heilbronn_convex": 11.5,
    },
    "BES": {
        "circle_packing_square": 18.6,
        "circle_packing_rect": 14.0,
        "heilbronn_convex": 13.7,
    },
}
