"""Paper metrics and ablation tables (Sec. 4)."""

from __future__ import annotations

from typing import Any


def table1_codec_comparison() -> list[dict[str, Any]]:
    return [
        {"codec": "TQCodec", "sr_khz": 44.1, "bitrates_kbps": "32,64,128", "codebooks": "5/10/20", "frame_rate": 689},
        {"codec": "DAC", "sr_khz": 44.1, "bitrates_kbps": "16", "codebooks": 18, "frame_rate": 86},
        {"codec": "Encodec", "sr_khz": "24/48", "bitrates_kbps": "24", "codebooks": 16, "frame_rate": "75/150"},
    ]


def table3_ablation() -> list[dict[str, Any]]:
    return [
        {"name": "DAC (16kbps)", "lsd": 0.927, "lsd_l": 0.877, "lsd_h": 1.016, "macs_e": "25G", "macs_d": "365G", "rf": 17706},
        {"name": "DAC-small (32kbps)", "lsd": 0.833, "lsd_l": 0.781, "lsd_h": 0.936, "macs_e": "25G", "macs_d": "365G", "rf": 17706},
        {"name": "Encodec + DAC RVQ", "lsd": 0.854, "lsd_l": 0.790, "lsd_h": 0.983, "macs_e": "2.4G", "macs_d": "6.31G", "rf": 2410},
        {"name": "+ RSimVQ", "lsd": 0.850, "lsd_l": 0.772, "lsd_h": 0.997, "macs_e": "2.4G", "macs_d": "6.31G", "rf": 2410},
        {"name": "+ Waveform loss", "lsd": 0.844, "lsd_l": 0.769, "lsd_h": 0.995, "macs_e": "2.4G", "macs_d": "6.31G", "rf": 2410},
        {"name": "+ Imbalanced Autoencoder", "lsd": 0.822, "lsd_l": 0.757, "lsd_h": 0.949, "macs_e": "72G", "macs_d": "6.31G", "rf": 2410},
        {"name": "64kbps", "lsd": 0.7715, "lsd_l": 0.647, "lsd_h": 0.950, "macs_e": "72G", "macs_d": "6.31G", "rf": 2410},
        {"name": "128kbps", "lsd": 0.6716, "lsd_l": 0.5166, "lsd_h": 0.9517, "macs_e": "72G", "macs_d": "6.31G", "rf": 2410},
    ]


def table4_subband_baseline() -> list[dict[str, Any]]:
    return [
        {"model": "Ogg-Vorbis", "bitrate_kbps": 48, "lsd_l": 0.763, "lsd_h": 2.847, "snr": 16.998},
        {"model": "TQCodec", "bitrate_kbps": 43, "lsd_l": 0.718, "lsd_h": 0.954, "snr": 16.074},
        {"model": "TQCodec + Subband", "bitrate_kbps": 43, "lsd_l": 0.702, "lsd_h": 1.163, "snr": 16.88},
    ]


def table5_subjective_mos() -> dict[str, float]:
    return {
        "mos_average": 4.18,
        "pass_rate_pct": 100.0,
        "excellent_rate_pct": 77.8,
        "preference_hq_ogg_pct": 39.4,
        "preference_aicodec_pct": 39.1,
        "no_significant_difference_pct": 21.5,
    }


def table2_datasets() -> list[dict[str, Any]]:
    return [
        {"name": "MusDBHQ", "tracks": 150},
        {"name": "Jingju", "tracks": 120},
        {"name": "Jamendo", "tracks": 55609},
        {"name": "Fma", "tracks": 106574},
        {"name": "Private dataset", "tracks": "100000+"},
    ]
