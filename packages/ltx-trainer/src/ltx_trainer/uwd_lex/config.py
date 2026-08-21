"""UWD lexicon evaluation — Malan et al., arXiv:2606.06183."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UwdLexConfig:
    paper_arxiv: str = "arXiv:2606.06183"
    title: str = (
        "Revisiting Lexicon Evaluation in Unsupervised Word Discovery"
    )
    framework: str = "UWD-Lex"

    # §2.2 domain-aware distillation scale analogue — clustering property focus
    domain_aware_alpha: float = 10.0  # not used; kept for cross-ref with USAD pattern

    # LibriSpeech dev-clean evaluation (§VI)
    eval_corpus: str = "LibriSpeech dev-clean"
    phoneme_aligner: str = "Montreal Forced Aligner"
    gt_phoneme_classes: int = 8372
    reference_k: int = 13_967

    # |K| sweep (Fig. 4)
    cluster_sizes_sweep: tuple[int, ...] = (500, 1000, 3000, 8372, 13_967, 20_000)

    # Fig. 4 — best lexicon by F1-WNES / d-PAcc
    best_system: str = "cosine_graph"
    best_k: int = 3000
    runner_up_system: str = "K→H"
    runner_up_k: int = 1000

    # Fig. 6 synthetic lexicons (|K|=|C|=8372), higher-is-better %
    synth_large_pure_nes_pct: float = 77.0
    synth_large_impure_nes_pct: float = 26.0
    synth_large_pure_wnes_pct: float = 46.0
    synth_large_impure_wnes_pct: float = 62.0
    synth_large_pure_ines_pct: float = 68.0
    synth_large_impure_ines_pct: float = 19.0
    synth_iwnes_gap_pct: float = 13.0  # iNES exaggerates vs iWNES

    metrics: tuple[str, ...] = (
        "NES",
        "WNES",
        "PAcc",
        "iNES",
        "iWNES",
        "iPAcc",
        "F1-WNES",
        "F1-NES",
        "d-PAcc",
        "Bitrate",
    )
    discovery_systems: tuple[str, ...] = (
        "K-Means++",
        "cosine_graph",
        "K→H",
    )
