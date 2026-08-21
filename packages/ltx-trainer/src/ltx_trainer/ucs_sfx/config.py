"""UCS SFX dataset unification — Beck & Lerch, DAFx26 / arXiv:2606.05571."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UcsSfxConfig:
    paper_arxiv: str = "arXiv:2606.05571"
    title: str = "Sound Effects Dataset Unification with the Universal Category System"
    framework: str = "UCS-SFX"
    venue: str = "DAFx26"

    ucs_version: str = "8.2.1"
    ucs_categories: int = 82
    ucs_subcategories: int = 453
    ucs_synonyms: int = 9972

    github_tools: str = "https://github.com/JunWooBeck/ucs-sfx-tools"
    github_envsound: str = "https://github.com/JunWooBeck/envsound-ucs"

    split_train: float = 0.70
    split_val: float = 0.15
    split_test: float = 0.15
    split_seed: int = 42
    split_corr_min: float = 0.99

    # Table 1 — conversion rates
    fsd50k_total: int = 51197
    fsd50k_classified_rate: float = 1.00
    fsd50k_ambiguous_pct: float = 15.8

    audioset_total: int = 33268
    audioset_classified: int = 32767
    audioset_classified_rate: float = 0.9849
    audioset_ambiguous_pct: float = 28.0

    esc50_total: int = 2000
    esc50_classified_rate: float = 1.00

    # EnvSound-UCS (Sec. 3.1)
    envsound_total: int = 58057
    envsound_categories: int = 59
    envsound_fsd50k: int = 33065
    envsound_audioset: int = 22992
    envsound_esc50: int = 2000

    # Table 4 — EnvSound macro F1 (SubCatflat self-trained)
    envsound_subcat_flat_f1: float = 0.49
    envsound_subcat_hier_oracle_f1: float = 0.73
    envsound_cat_f1: float = 0.46

    # Table 3 anchors
    fsd50k_cat_f1: float = 0.52
    esc50_cat_f1: float = 0.89
    audioset_cat_f1: float = 0.42

    source_datasets: tuple[str, ...] = ("FSD50K", "AudioSet", "ESC-50")
