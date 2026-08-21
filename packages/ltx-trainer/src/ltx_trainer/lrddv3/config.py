"""LRDDv3 long-range drone detection dataset (Peterson et al., arXiv:2605.25942)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LRDDv3Config:
    """Defaults from paper Sec. III–IV."""

    rgb_resolution: tuple[int, int] = (3840, 2160)  # 4K
    ir_resolution: tuple[int, int] = (640, 512)
    sample_fps: int = 5
    source_fps: int = 30

    num_rgb_images: int = 102532
    num_rgb_with_range: int = 101227
    num_rgb_no_range: int = 1305
    num_ir_images: int = 29630
    num_video_clips: int = 128
    video_hours: float = 5.7
    collection_days: int = 17
    collection_months: int = 8

    range_min_m: float = 0.0
    range_max_m: float = 200.0
    min_bbox_pixels: int = 12

    test_rgb_images: int = 20000
    test_video_clips: int = 34

    annotations_drone: int = 93652
    annotations_bird: int = 6031
    annotations_airplane: int = 722

    earth_radius_m: float = 6_371_000.0
    benchmark_detector: str = "YOLOv11m"
    benchmark_train_epochs: int = 50
    benchmark_input_sizes: tuple[int, ...] = (640, 1920)

    dataset_url: str = "https://research.coe.drexel.edu/ece/imaple/lrddv3/"

    weather_counts: dict[str, int] = field(
        default_factory=lambda: {"clear": 76965, "rainy": 24356, "snowy": 1211}
    )
    lighting_counts: dict[str, int] = field(
        default_factory=lambda: {
            "sunny": 33334,
            "partly_cloudy": 28183,
            "cloudy": 41015,
        }
    )
