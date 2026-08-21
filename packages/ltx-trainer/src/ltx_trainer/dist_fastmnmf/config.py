"""Distributed FastMNMF configuration (arXiv:2605.19388, APSIPA ASC 2026)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistFastmnmfConfig:
    paper_id: str = "2605.19388"
    paper_url: str = "https://arxiv.org/abs/2605.19388"
    title: str = (
        "Fast Multichannel NMF with Block-Diagonal Spatial Covariance Matrices "
        "for Efficient Blind Source Separation Using Distributed Microphone Arrays"
    )
    venue: str = "APSIPA ASC 2026"

    # Simulation (§IV-A)
    n_subarrays_l: int = 3
    mics_per_subarray: int = 4
    room_m: tuple[float, float, float] = (6.0, 4.0, 2.5)
    rt60_ms: float = 300.0
    fs_hz: int = 16_000
    stft_win_ms: float = 256.0
    stft_hop_ms: float = 64.0

    # FastMNMF hyper-parameters
    nmf_bases_k: int = 16
    n_iterations: int = 200
    denominator_floor: float = 1e-6

    # Experiment counts
    n_mixtures: int = 120
    n_nmf_inits: int = 10

    @property
    def total_mics_m(self) -> int:
        return self.n_subarrays_l * self.mics_per_subarray


@dataclass(frozen=True)
class SdrResults:
    """Fig. 2 / §IV-B mean SDR improvement (dB)."""

    three_source: dict[str, float]
    five_source: dict[str, float]


@dataclass(frozen=True)
class TimingResults:
    """Table II — 10 s mixture, 200 iterations, Ryzen 5 5600X single thread."""

    fastmnmf_one_subarray_s: float = 109.3
    fastmnmf_all_subarrays_s: float = 694.0
    distributed_fastmnmf_s: float = 235.3
