"""Thread E: CLIP zero-shot weather stub (Sec. III-H)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.cadenet.schema import WeatherCondition

WEATHER_PROMPTS: dict[WeatherCondition, str] = {
    WeatherCondition.RAIN: "a photo of rain on a road",
    WeatherCondition.FOG: "a photo of foggy weather",
    WeatherCondition.SAND: "a photo of sandstorm or dusty weather",
    WeatherCondition.SNOW: "a photo of snowy weather",
    WeatherCondition.CLEAR: "a photo of clear weather",
}


class CLIPWeatherStub(nn.Module):
    """ViT-B/32 stand-in for zero-shot weather classification."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.image_enc = nn.Sequential(
            nn.Conv2d(3, embed_dim, 8, stride=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.text_embed = nn.Embedding(len(WEATHER_PROMPTS), embed_dim)
        labels = list(WEATHER_PROMPTS.keys())
        for i, lab in enumerate(labels):
            torch.nn.init.normal_(self.text_embed.weight[i], std=0.02)

    def classify(self, frame: Tensor) -> WeatherCondition:
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        img = F.normalize(self.image_enc(frame), dim=-1)
        text = F.normalize(self.text_embed.weight, dim=-1)
        sim = img @ text.T
        idx = int(sim.argmax(dim=-1).item())
        return list(WEATHER_PROMPTS.keys())[idx]
