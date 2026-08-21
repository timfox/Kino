"""Infant cry ACC–MIC cross-modal validation (arXiv:2605.28687)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CryLabel(str, Enum):
    CRY_ONLY = "cry_only"
    CRY_NOISE = "cry_noise"
    NON_CRY = "non_cry"


class VocalMeasure(str, Enum):
    F0 = "F0"
    JCV = "JCV"
    JLOCAL = "Jlocal"
    SCV = "SCV"
    SLOCAL = "Slocal"
    CPP = "CPP"
    HNR = "HNR"


@dataclass
class CryAccConfig:
    paper_arxiv: str = "arXiv:2605.28687"
    venue: str = "IEEE TASLP"
    irb: str = "IRB-P00038922"
    # Sample
    n_infants_total: int = 85
    n_4_month: int = 41
    n_12_month: int = 44
    # Recording
    acc_sample_rate_hz: float = 11_025.0
    mic_sample_rate_hz: float = 44_100.0
    acc_sensor: str = "Knowles BU-27135"
    window_ms: float = 50.0
    rms_exclude_threshold: float = 0.01
    f0_floor_hz: float = 200.0
    f0_ceiling_hz: float = 1500.0
    segments_per_infant: int = 20
    # Table I ICC(A,1) overall anchors
    icc_f0: float = 0.947
    icc_jcv: float = 0.949
    icc_jlocal: float = 0.873
    icc_scv: float = 0.187
    icc_slocal: float = 0.322
    icc_cpp: float = 0.583
    icc_hnr: float = 0.411
    # Table II bias (ACC − MIC) 4-month anchors
    bias_scv_4m_pp: float = -5.983
    bias_slocal_4m_pp: float = -2.803
    bias_cpp_4m_db: float = -0.079
    bias_hnr_4m_db: float = 4.381
