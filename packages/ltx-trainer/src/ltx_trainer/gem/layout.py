"""Pipeline stages and limitations for GEM stub."""

from __future__ import annotations

PIPELINE_STAGES = (
    "embed_and_normalize",
    "seed_corpus_sample",
    "mm_vmf_teacher",
    "gis_representatives",
    "taxonomy_labels",
    "fasttext_student_distill",
    "mixing_doremi_regmix_perf",
)

LIMITATIONS = (
    "Reference stub: toy MM on synthetic hypersphere embeddings only.",
    "No full CommonCrawl pipeline or FastText training harness.",
    "Bessel normalization uses low-d approximation; not web-scale.",
    "Mixing weights (DoReMi/RegMix/Perf) are table excerpts, not fitted.",
)
