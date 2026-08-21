"""Training strategies for different conditioning modes.

This package implements the Strategy Pattern to handle different training modes:
- Text-to-video training (standard generation, optionally with audio) [DEPRECATED]
- Video-to-video training (IC-LoRA mode with reference videos) [DEPRECATED]
- Flexible training (unified conditioning framework supporting all scenarios) [RECOMMENDED]
- Audio reference-only IC-LoRA (Gopex/Kino extension for speaker identity without ref video)

Each strategy encapsulates the specific logic for preparing model inputs and computing loss.
"""

import warnings

from ltx_trainer import logger
from ltx_trainer.training_strategies.audio_ref_only_ic import AudioRefOnlyICConfig, AudioRefOnlyICStrategy
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer.training_strategies.flexible import FlexibleStrategy, FlexibleStrategyConfig
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig, VideoToVideoStrategy

# Type alias for all strategy config types
TrainingStrategyConfig = (
    TextToVideoConfig | VideoToVideoConfig | FlexibleStrategyConfig | AudioRefOnlyICConfig
)

__all__ = [
    "DEFAULT_FPS",
    "VIDEO_SCALE_FACTORS",
    "AudioRefOnlyICConfig",
    "AudioRefOnlyICStrategy",
    "FlexibleStrategy",
    "FlexibleStrategyConfig",
    "ModelInputs",
    "TextToVideoConfig",
    "TextToVideoStrategy",
    "TrainingStrategy",
    "TrainingStrategyConfig",
    "TrainingStrategyConfigBase",
    "VideoToVideoConfig",
    "VideoToVideoStrategy",
    "get_training_strategy",
    "strategy_requires_audio",
]


def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.

    The strategy is determined by the `name` field in the configuration.

    Args:
        config: Strategy-specific configuration with a `name` field

    Returns:
        The appropriate training strategy instance

    Raises:
        ValueError: If strategy name is not supported
    """

    match config:
        case TextToVideoConfig():
            warnings.warn(
                "The 'text_to_video' training strategy is deprecated and will be removed "
                "in a future version. Please migrate to the 'flexible' strategy. "
                "See the migration guide in the documentation.",
                DeprecationWarning,
                stacklevel=2,
            )
            strategy = TextToVideoStrategy(config)
        case VideoToVideoConfig():
            warnings.warn(
                "The 'video_to_video' training strategy is deprecated and will be removed "
                "in a future version. Please migrate to the 'flexible' strategy. "
                "See the migration guide in the documentation.",
                DeprecationWarning,
                stacklevel=2,
            )
            strategy = VideoToVideoStrategy(config)
        case FlexibleStrategyConfig():
            strategy = FlexibleStrategy(config)
        case AudioRefOnlyICConfig():
            strategy = AudioRefOnlyICStrategy(config)
        case _:
            raise ValueError(f"Unknown training strategy config type: {type(config).__name__}")

    if hasattr(config, "with_audio"):
        audio_mode = "(audio enabled 🔈)" if config.with_audio else "(audio disabled 🔇)"
    elif hasattr(config, "audio") and config.audio is not None:
        audio_mode = "(audio enabled 🔈)"
    elif config.name == "audio_ref_only_ic":
        audio_mode = "(audio enabled 🔈)"
    else:
        audio_mode = "(audio disabled 🔇)"

    logger.debug(f"🎯 Using {strategy.__class__.__name__} training strategy {audio_mode}")
    return strategy


def strategy_requires_audio(config: TrainingStrategyConfig) -> bool:
    """Whether training should load audio VAE / train the audio branch."""
    if isinstance(config, AudioRefOnlyICConfig):
        return True
    if isinstance(config, TextToVideoConfig):
        return config.with_audio
    if isinstance(config, FlexibleStrategyConfig):
        return config.audio is not None
    return False
