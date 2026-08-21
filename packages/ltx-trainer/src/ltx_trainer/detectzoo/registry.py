"""Detector and dataset registry (Tables 2–5)."""

from __future__ import annotations

from typing import Literal

Modality = Literal["text", "image", "audio"]

# Table 2 — 36 text detectors
TEXT_DETECTORS: tuple[str, ...] = (
    "log_likelihood",
    "log_rank",
    "rank",
    "entropy",
    "lrr",
    "lastde",
    "gecscore",
    "biscope",
    "detectgpt",
    "fast_detectgpt",
    "adadetectgpt",
    "npr",
    "lastde_pp",
    "glimpse",
    "binoculars",
    "dna_gpt",
    "dna_detectllm",
    "revise_detect",
    "raidar",
    "ghostbuster",
    "tocsin",
    "ipad",
    "text_fluoroscopy",
    "coco",
    "phd",
    "mle_ide",
    "roberta_base",
    "roberta_large",
    "radar",
    "imbd",
    "remodetect",
    "detective",
    "irm",
    "dsvdd",
    "hrn",
    "energy_detector",
)

# Table 3 — 15 image detectors
IMAGE_DETECTORS: tuple[str, ...] = (
    "aeroblade",
    "aide",
    "cnnspot",
    "c2p_clip",
    "cospy",
    "d3",
    "drct",
    "fatformer",
    "freqnet",
    "lgrad",
    "manifold_bias",
    "npr_deepfake",
    "patchcraft",
    "safe",
    "univfd",
)

# Table 4 — 10 audio detectors
AUDIO_DETECTORS: tuple[str, ...] = (
    "rawnet2",
    "aasist",
    "rawgat_st",
    "res_tssdnet",
    "samo",
    "ast_asvspoof",
    "anti_deepfake_wav2vec",
    "anti_deepfake_hubert",
    "anti_deepfake_xlsr2b",
    "xlsr_sls",
)

ALL_DETECTORS: dict[Modality, tuple[str, ...]] = {
    "text": TEXT_DETECTORS,
    "image": IMAGE_DETECTORS,
    "audio": AUDIO_DETECTORS,
}

_DETECTOR_INDEX: dict[str, Modality] = {}
for _mod, _names in ALL_DETECTORS.items():
    for _n in _names:
        _DETECTOR_INDEX[_n] = _mod

# Table 5 — 22 benchmark datasets (representative registry keys)
DATASETS: dict[Modality, tuple[str, ...]] = {
    "text": (
        "hc3",
        "hc3_plus",
        "cheat",
        "openllmtext",
        "mage",
        "m4",
        "raid",
        "l2r",
        "turingbench",
        "writingprompts",
        "xsum",
    ),
    "image": (
        "forensynths",
        "self_synthesis",
        "ufd",
        "aigcdetect",
        "genimage",
        "drct_2m",
        "chameleon",
    ),
    "audio": (
        "asvspoof2019",
        "for_corpus",
        "in_the_wild",
        "deepfake_eval_2024",
    ),
}


def modality_for(name: str) -> Modality:
    key = name.lower().strip().replace("-", "_")
    if key not in _DETECTOR_INDEX:
        raise KeyError(f"unknown detector: {name}")
    return _DETECTOR_INDEX[key]


def total_detectors() -> int:
    return sum(len(v) for v in ALL_DETECTORS.values())


def total_datasets() -> int:
    return sum(len(v) for v in DATASETS.values())
