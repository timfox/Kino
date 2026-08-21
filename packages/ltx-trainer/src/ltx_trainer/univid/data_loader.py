"""Data loader for custom datasets (e.g., .pt files with pre-computed embeddings)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from ltx_trainer.univid.moderation import VideoSignals


@dataclass
class DataLoaderConfig:
    """Configuration for custom dataset loading."""
    data_dir: str = "data"
    file_pattern: str = "*.pt"
    caption_source: str = "filename"  # Options: "filename", "metadata", "custom"


class CustomDatasetLoader:
    """Loader for custom datasets with pre-computed embeddings."""
    
    def __init__(self, config: DataLoaderConfig | None = None):
        self.config = config or DataLoaderConfig()
        self.data_dir = Path(self.config.data_dir)
        
    def get_pt_files(self) -> list[Path]:
        """Get all .pt files in the data directory."""
        return sorted(self.data_dir.glob(self.config.file_pattern))
    
    def load_single_file(self, filepath: Path) -> VideoSignals:
        """Load a single .pt file and create VideoSignals."""
        # Load the tensor data
        data = torch.load(filepath, weights_only=False)
        
        # Use filename as video_id (without extension)
        video_id = filepath.stem
        
        # For now, use a placeholder caption based on filename
        # In production, you could:
        # 1. Extract caption from metadata
        # 2. Use an actual VLM to generate captions
        # 3. Use a custom captioning function
        caption = self._generate_caption(filepath, data)
        
        return VideoSignals(
            video_id=video_id,
            caption=caption,
            fusion_embedding=None  # Will be computed by FusionNetwork
        )
    
    def _generate_caption(self, filepath: Path, data: dict) -> str:
        """Generate a caption for the video."""
        if self.config.caption_source == "filename":
            # Use filename as caption (replace underscores with spaces)
            return filepath.stem.replace("_", " ")
        elif self.config.caption_source == "metadata":
            # Look for caption in metadata (if available)
            if "caption" in data:
                return data["caption"]
            elif "prompt" in data:
                return data["prompt"]
            else:
                return filepath.stem.replace("_", " ")
        else:
            # Fallback to filename
            return filepath.stem.replace("_", " ")
    
    def load_all(self) -> list[VideoSignals]:
        """Load all .pt files and return list of VideoSignals."""
        files = self.get_pt_files()
        signals = []
        for filepath in files:
            try:
                signals.append(self.load_single_file(filepath))
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
        return signals
    
    def load_batch(self, batch_size: int = 10) -> list[VideoSignals]:
        """Load a batch of files."""
        files = self.get_pt_files()
        batch = []
        for filepath in files[:batch_size]:
            try:
                batch.append(self.load_single_file(filepath))
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
        return batch


def load_custom_dataset(
    data_dir: str = "data",
    file_pattern: str = "*.pt",
    caption_source: str = "filename"
) -> list[VideoSignals]:
    """Convenience function to load custom dataset."""
    config = DataLoaderConfig(
        data_dir=data_dir,
        file_pattern=file_pattern,
        caption_source=caption_source
    )
    loader = CustomDatasetLoader(config)
    return loader.load_all()


def load_custom_dataset_batch(
    data_dir: str = "data",
    file_pattern: str = "*.pt",
    caption_source: str = "filename",
    batch_size: int = 10
) -> list[VideoSignals]:
    """Convenience function to load a batch of custom dataset."""
    config = DataLoaderConfig(
        data_dir=data_dir,
        file_pattern=file_pattern,
        caption_source=caption_source
    )
    loader = CustomDatasetLoader(config)
    return loader.load_batch(batch_size)