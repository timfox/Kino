"""LC-DeepBeam framework card + paper excerpt tables (arXiv:2605.21141)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lc_deepbeam.config import LcDeepBeamConfig
from ltx_trainer.lc_deepbeam.layout import LIMITATIONS
from ltx_trainer.lc_deepbeam.mock import evaluation_smoke


def framework_card(cfg: LcDeepBeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LcDeepBeamConfig()
    return {
        "name": "Linearly Constrained Deep Beamformer (LC-DeepBeam) for multi-speaker scenarios",
        "paper": cfg.paper_arxiv,
        "repo": cfg.repo,
        "authors": "Ilai Zaidel, Ori Engel, Bar Engel, Sharon Gannot (Bar-Ilan University)",
        "problem": "Learn beamforming weights from multichannel mixtures while enforcing LCMV-like linear constraints (pass + null).",
        "core_idea": (
            "Train a DNN beamformer with an augmented-Lagrangian inspired multi-term loss: "
            "reconstruction (SI-SDR) + distortionless pass constraint + interference-subspace null penalty."
        ),
        "spatial_guidance": {
            "target_rtf": "Covariance whitening (CW) RTF estimate from target-only frames (Eqs. 6–10).",
            "interference_subspace": "Dominant eigensubspace from interference-only frames (Eq. 11).",
        },
        "baseline": "Analytical LCMV beamformer built from the same estimated spatial signatures (Eq. 13).",
        "loss_eq12": {
            "recon": "-SI-SDR(ŝ, s_target)",
            "pass": "λ_pass E_k[|w^H a_target - 1|^2]",
            "null": "λ_null E_k[10 log10(||w^H A_interf||^2 + ε)]",
            "schedule": "λ_pass, λ_null ramped after 10-epoch warmup (Sec. 3.3).",
        },
        "dataset_sim": {
            "array": "8-mic linear array, random tilt ±45°",
            "speakers": "J ∈ {2,3} from LibriSpeech; stationary babble background",
            "segments": "0.5s noise-only, 1s target-only, 1s interferer-only, 1.5s full mixture (estimation), then 4s full mixture (eval).",
            "train_set": "20,000 multichannel recordings",
        },
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }


def table1_three_speaker_anechoic() -> dict[str, Any]:
    """Table 1 excerpt — three-speaker scenario (anechoic target/interference)."""
    return {
        "scenario": "J=3, anechoic target/interference",
        "si_sdr_db": {"Input": -4.65, "Estimated RTF": 0.63, "No RTF": 0.62, "Oracle RTF": 1.04, "LCMV": -1.94},
        "snr_db": {"Input": 1.46, "Estimated RTF": 5.74, "No RTF": 6.16, "Oracle RTF": 6.02, "LCMV": 2.96},
        "sir_db": {"Input": -3.39, "Estimated RTF": 4.90, "No RTF": 5.15, "Oracle RTF": 5.49, "LCMV": 6.70},
        "pwr_ratio_db": {
            "interferer1": {"Estimated RTF": -10.18, "No RTF": -10.69, "Oracle RTF": -10.89, "LCMV": -10.31},
            "interferer2": {"Estimated RTF": -8.53, "No RTF": -9.02, "Oracle RTF": -9.58, "LCMV": -9.96},
            "noise": {"Estimated RTF": -4.28, "No RTF": -4.69, "Oracle RTF": -4.56, "LCMV": -1.50},
        },
    }


def table2_two_speaker_reverberant() -> dict[str, Any]:
    """Table 2 excerpt — two-speaker scenario (reverberant target/interference)."""
    return {
        "scenario": "J=2, reverberant target/interference",
        "si_sdr_db": {"Input": -1.81, "Estimated RTF": 0.33, "No RTF": 0.05, "Oracle RTF": 0.40, "LCMV": -3.50},
        "snr_db": {"Input": 3.30, "Estimated RTF": 5.61, "No RTF": 6.33, "Oracle RTF": 6.11, "LCMV": 5.24},
        "sir_db": {"Input": -0.03, "Estimated RTF": 4.78, "No RTF": 4.62, "Oracle RTF": 5.00, "LCMV": 5.58},
        "pwr_ratio_db": {
            "interferer1": {"Estimated RTF": -4.81, "No RTF": -4.66, "Oracle RTF": -5.03, "LCMV": -5.61},
            "noise": {"Estimated RTF": -2.31, "No RTF": -3.03, "Oracle RTF": -2.81, "LCMV": -1.94},
        },
    }


def table3_fully_overlapped_conceptual() -> dict[str, Any]:
    """Table 3 excerpt — fully overlapped (conceptual; CW requires separated activity)."""
    return {
        "scenario": "J=3, anechoic, fully overlapped (conceptual)",
        "si_sdr_db": {"Input": -4.65, "Oracle RTF": 1.28, "No RTF": -4.62},
        "snr_db": {"Input": 1.46, "Oracle RTF": 5.85, "No RTF": 1.52},
        "sir_db": {"Input": -3.39, "Oracle RTF": 5.74, "No RTF": -3.34},
        "pwr_ratio_db": {
            "interferer1": {"Oracle RTF": -10.91, "No RTF": -0.02},
            "interferer2": {"Oracle RTF": -9.81, "No RTF": -0.04},
            "noise": {"Oracle RTF": -4.39, "No RTF": -0.05},
        },
    }


def headline_results(cfg: LcDeepBeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LcDeepBeamConfig()
    return {
        "finding": (
            "DNN beamformer with explicit pass + null constraints outperforms analytical LCMV built from the same "
            "estimated spatial signatures (higher SI-SDR/SNR, better noise attenuation) and yields more controlled sidelobes."
        ),
        "table1_input_si_sdr_db": cfg.table1_three_spk_anechoic_input_si_sdr,
        "table1_est_rtf_si_sdr_db": cfg.table1_three_spk_anechoic_est_rtf_si_sdr,
        "table2_est_rtf_si_sdr_db": cfg.table2_two_spk_reverb_est_rtf_si_sdr,
        "table3_oracle_vs_no_rtf_si_sdr_db": (
            cfg.table3_fully_overlapped_oracle_si_sdr,
            cfg.table3_fully_overlapped_no_rtf_si_sdr,
        ),
    }


def evaluation_demo(cfg: LcDeepBeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LcDeepBeamConfig()
    return {"paper": cfg.paper_arxiv, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: LcDeepBeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LcDeepBeamConfig()
    return {
        "framework": framework_card(cfg),
        "table1_three_speaker_anechoic": table1_three_speaker_anechoic(),
        "table2_two_speaker_reverberant": table2_two_speaker_reverberant(),
        "table3_fully_overlapped": table3_fully_overlapped_conceptual(),
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }

