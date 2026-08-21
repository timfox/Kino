"""Config for LC-DeepBeam stub (arXiv:2605.21141)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LcDeepBeamConfig:
    paper_arxiv: str = "arXiv:2605.21141"
    repo: str = "https://github.com/GannotLab/LC-DeepBeam"

    # Array / scenario (Sec. 4.1)
    microphones_m: int = 8
    speakers_j_choices: tuple[int, ...] = (2, 3)
    fs_hz: int = 16000
    recording_seconds: float = 8.0
    estimation_segment_seconds: float = 4.0

    # Loss weights / schedule (Sec. 3.3)
    warmup_epochs: int = 10
    lambda_pass_final: float = 1.0
    lambda_null_final: float = 1.0
    null_eps: float = 1e-8

    # Table headline numbers (Tables 1–3)
    table1_three_spk_anechoic_input_si_sdr: float = -4.65
    table1_three_spk_anechoic_est_rtf_si_sdr: float = 0.63
    table1_three_spk_anechoic_lcmv_si_sdr: float = -1.94

    table2_two_spk_reverb_input_si_sdr: float = -1.81
    table2_two_spk_reverb_est_rtf_si_sdr: float = 0.33
    table2_two_spk_reverb_lcmv_si_sdr: float = -3.50

    table3_fully_overlapped_oracle_si_sdr: float = 1.28
    table3_fully_overlapped_no_rtf_si_sdr: float = -4.62
