"""QoMEX VSR video quality study (Herb et al., arXiv:2605.25940)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VSRVQAConfig:
    """Defaults from paper Sec. II–IV."""

    conference: str = "QoMEX 2026"
    paper_arxiv: str = "arXiv:2605.25940"

    num_source_clips: int = 6
    num_participants: int = 32
    num_participants_valid: int = 28
    num_pvs: int = 222
    viewing_distance: str = "1.5H"
    rating_scale: int = 5  # ACR

    output_resolution: tuple[int, int] = (3840, 2160)
    output_fps: int = 60
    clip_duration_s: tuple[int, int] = (8, 10)

    upscale_methods: tuple[str, ...] = (
        "Lanczos",
        "Rhea",
        "SCST",
        "DOVE",
        "SeedVR2",
        "Starlight Mini",
    )
    codecs: tuple[str, ...] = ("uncompressed", "AV1", "DCVC-RT")
    source_resolutions: tuple[str, ...] = ("360p", "720p")

    sos_a: float = 0.254
    outlier_plcc_threshold: float = 0.70

    dataset_repo: str = (
        "https://github.com/Telecommunication-Telemedia-Assessment/AVT-VQDB-UHD-1-VSR"
    )

    processing_seconds_per_frame: dict[str, float] = field(
        default_factory=lambda: {
            "SCST": 96.0,
            "DOVE": 18.0,
            "SeedVR2": 11.0,
        }
    )
