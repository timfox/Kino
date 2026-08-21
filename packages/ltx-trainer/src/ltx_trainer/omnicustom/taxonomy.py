"""Customization setting taxonomy (paper Table 1)."""

from __future__ import annotations

from enum import Enum


class CustomizationSetting(str, Enum):
    TYPICAL_VIDEO = "typical_video_customization"
    AUDIO_DRIVEN = "audio_driven_video_customization"
    SYNC_AV = "sync_audio_video_customization"


SETTING_TRAITS: dict[CustomizationSetting, dict[str, bool]] = {
    CustomizationSetting.TYPICAL_VIDEO: {
        "identity_preservation": True,
        "audio_containing": False,
        "audio_customization": False,
        "background_sounds": False,
    },
    CustomizationSetting.AUDIO_DRIVEN: {
        "identity_preservation": True,
        "audio_containing": True,
        "audio_customization": False,
        "background_sounds": False,
    },
    CustomizationSetting.SYNC_AV: {
        "identity_preservation": True,
        "audio_containing": True,
        "audio_customization": True,
        "background_sounds": True,
    },
}
