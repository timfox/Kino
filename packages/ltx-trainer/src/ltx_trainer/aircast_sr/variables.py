"""Variable names and channel layout for AirCast-SR."""

from __future__ import annotations

TARGET_VARIABLES: tuple[str, ...] = (
    "precipitation",
    "t2m",
    "q2m",
    "u10",
    "v10",
    "sp",
    "dlwrf",
)

# 17 GraphCast atmospheric fields × 3 pressure levels + topography + sky-view + cos(sza)
CONDITIONING_STATIC: tuple[str, ...] = ("topography", "sky_view_factor", "cos_solar_zenith")
# Paper: 17 GraphCast fields (incl. three pressure levels) + 3 static = 20 channels
CONDITIONING_ATMOSPHERIC: tuple[str, ...] = tuple(f"graphcast_{i:02d}" for i in range(17))

CONDITIONING_CHANNELS: tuple[str, ...] = CONDITIONING_ATMOSPHERIC + CONDITIONING_STATIC

assert len(CONDITIONING_CHANNELS) == 20


def channel_spec() -> dict[str, object]:
    return {
        "target_count": len(TARGET_VARIABLES),
        "conditioning_count": len(CONDITIONING_CHANNELS),
        "denoiser_in": 7 + 20 + 1,  # noisy target + cond + optional lead-time plane
        "denoiser_out": 7,
        "targets": list(TARGET_VARIABLES),
        "conditioning": list(CONDITIONING_CHANNELS),
    }
