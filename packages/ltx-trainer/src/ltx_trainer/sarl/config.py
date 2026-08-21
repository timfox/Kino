"""SARL spatial audio probing benchmark — Chen et al., arXiv:2606.05544."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SarlConfig:
    paper_arxiv: str = "arXiv:2606.05544"
    title: str = "Probing Spatial Structure in Pretrained Audio Representations"
    framework: str = "SARL"
    venue: str = "arXiv / NYU MARL"

    sample_rate_hz: int = 24000
    clip_seconds: float = 10.0
    event_classes: int = 7
    clips_per_class: int = 4000

    source_rir_train: int = 12000
    room_rir_train: int = 12000
    scenes_per_epoch_train: int = 15000

    probe_lr: float = 1e-4
    probe_epochs: int = 20
    probe_seeds: int = 3

    azimuth_bins: int = 36
    elevation_bins: int = 12
    distance_bins: int = 20
    rt60_bins: int = 29
    volume_bins: int = 5
    shape_classes: int = 4

    num_encoders: int = 13

    # Fig. 2 aggregated baseline-normalized improvement anchors
    gram_f_semantic: float = 0.86
    gram_f_localization: float = 0.78
    gram_f_room: float = 0.48

    a_mae_semantic: float = 0.82
    a_mae_localization: float = 0.62
    a_mae_room: float = 0.55

    seld_s_localization: float = 0.84
    seld_s_room: float = 0.18

    encodec_room: float = 0.12
    encodec_localization: float = 0.22

    # Fig. 3 sensitivity anchors (source > room)
    gram_f_source_sensitivity: float = 0.38
    gram_f_room_sensitivity: float = 0.21
    sfd_source_sensitivity: float = 0.72
    sfd_room_sensitivity: float = 0.55
