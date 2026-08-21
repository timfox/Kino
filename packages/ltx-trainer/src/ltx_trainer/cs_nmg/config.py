"""CS-NMG — POI-aware contrastive CS-ASR with LLM near-misses (arXiv:2606.06985)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CsNmgConfig:
    paper_arxiv: str = "arXiv:2606.06985"
    title: str = (
        "Contrastive Training with LLM-generated Near-Misses for Robust "
        "Code-Switching Speech Recognition"
    )
    framework: str = "CS-NMG"
    backbone: str = "Whisper-small + LoRA (r=16, α=32, dropout=0.05)"

    # Datasets (§5.1)
    datasets: tuple[str, ...] = ("CS-FLEURS cmn-eng", "ViMedCSS vie-eng")

    # N-best / near-miss generation (§3, §5.1)
    n_best: int = 10
    poi_neighborhood_r: int = 1
    near_miss_k: int = 5
    acoustic_margin_delta: float = 4.0
    tau_text: float = 0.4
    tau_phoneme: float = 0.6
    llm: str = "Gemini 2.5 Pro (offline POI expansion)"

    # Training (§4, §5.1)
    lambda_cl: float = 0.1
    infonce_beta: float = 1.0
    alpha_wce_cmn: float = 1.7
    alpha_wce_vie: float = 2.0

    # Table 2 — WCE + CL tri-level (best row)
    cmn_wer: float = 14.06
    cmn_pier: float = 15.10
    vie_wer: float = 21.87
    vie_pier: float = 18.74

    # Table 2 — CE baseline
    cmn_wer_ce: float = 16.67
    cmn_pier_ce: float = 17.25
    vie_wer_ce: float = 24.72
    vie_pier_ce: float = 21.95

    # Table 2 — WCE + CL N-best only
    cmn_wer_nb_cl: float = 14.93
    cmn_pier_nb_cl: float = 15.72
    vie_wer_nb_cl: float = 22.86
    vie_pier_nb_cl: float = 19.10

    # Table 3 — tri-level gate NM/utt
    cmn_nm_per_utt: float = 3.77
    vie_nm_per_utt: float = 3.81

    filter_variants: tuple[str, ...] = (
        "No filter",
        "Acoustic",
        "Ac.+Text",
        "Ac.+Ph.",
        "Ac.+Ph.+Text",
    )
