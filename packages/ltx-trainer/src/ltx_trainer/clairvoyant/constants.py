"""Clairvoyant predictive SJF scheduling constants (Sundaresan, arXiv:2606.07248)."""

from __future__ import annotations

PAPER_ARXIV = "2606.07248"
PAPER_TITLE = (
    "Clairvoyant: Predictive SJF Scheduling to Mitigate Head-of-Line Blocking in Serial LLM Backends"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
UPSTREAM_REPO = "https://github.com/Aravind0403/clairvoyant-scheduler"

# Response-length class boundaries (Llama-2 tokenizer token counts)
SHORT_MAX_TOKENS = 200
MEDIUM_MAX_TOKENS = 800  # Long is >= MEDIUM_MAX_TOKENS

# Starvation timeout heuristic (Sec. 3.4)
STARVATION_TAU_MULTIPLIER = 3.0

# Paper-reported predictor latency (ms, Apple M1 CPU)
PREDICTOR_LATENCY_MS_SHAREGPT = 0.029
PREDICTOR_LATENCY_MS_LMSYS = 0.015

# Deployment utilisation band where SJF helps (Sec. 5.4)
UTIL_BENEFIT_RHO_MIN = 0.55
UTIL_BENEFIT_RHO_MAX = 0.80
UTIL_PEAK_RHO = 0.74

# 13 instruction-verb categories (Sec. 3.2)
INSTRUCTION_VERBS = (
    "what",
    "write",
    "explain",
    "summarize",
    "how",
    "list",
    "implement",
    "compare",
    "describe",
    "generate",
    "why",
    "define",
    "other",
)

CODE_KEYWORDS = (
    "function",
    "class",
    "implement",
    "algorithm",
    "code",
    "python",
    "javascript",
    "debug",
    "compile",
    "api",
    "sql",
    "regex",
)

LENGTH_CONSTRAINT_KEYWORDS = (
    "brief",
    "briefly",
    "concise",
    "detailed",
    "in one sentence",
    "one paragraph",
    "short answer",
    "long answer",
    "keep it short",
    "be concise",
)

FORMAT_KEYWORDS = (
    "table",
    "list",
    "json",
    "csv",
    "markdown",
    "bullet",
    "outline",
    "schema",
)

CLAUSE_MARKERS = (
    " because ",
    " although ",
    " though ",
    " while ",
    " whereas ",
    " if ",
    " when ",
    " where ",
    " which ",
    " that ",
    " who ",
    " whom ",
    " whose ",
)
