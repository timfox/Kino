"""Unified load_detector / predict API (Listing 1 stub)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ltx_trainer.detectzoo.config import DetectZooConfig
from ltx_trainer.detectzoo.registry import modality_for
from ltx_trainer.detectzoo.result import DetectionResult, Label


class BaseDetector(ABC):
    """Shared detector interface."""

    name: str
    modality: str

    def __init__(self, cfg: DetectZooConfig | None = None, **kwargs: Any) -> None:
        self.cfg = cfg or DetectZooConfig()
        self.device = kwargs.get("device", "cpu")
        self.threshold = float(kwargs.get("threshold", self.cfg.default_threshold))

    @abstractmethod
    def predict(self, sample: Any) -> DetectionResult:
        raise NotImplementedError


class TextDetectorStub(BaseDetector):
    modality = "text"

    def predict(self, sample: str) -> DetectionResult:
        text = str(sample)
        tokens = text.split()
        if not tokens:
            score = 0.0
        else:
            # stub: low lexical diversity -> higher AI score
            uniq = len(set(tokens)) / len(tokens)
            score = 1.0 - uniq + 0.1 * (1.0 if len(text) > 400 else 0.0)
        label: Label = "ai" if score >= self.threshold else "human"
        conf = 1.0 / (1.0 + math.exp(-abs(score - self.threshold) * 4))
        return DetectionResult(score=score, label=label, confidence=conf, metadata={"detector": self.name})


class ImageDetectorStub(BaseDetector):
    modality = "image"

    def predict(self, sample: str | Path) -> DetectionResult:
        # stub without loading PIL: path-based hash proxy
        p = str(sample)
        h = sum(ord(c) for c in p[-64:]) % 1000
        score = (h / 1000.0) * 0.6 + 0.2
        label: Label = "ai" if score >= self.threshold else "human"
        conf = min(1.0, abs(score - 0.5) + 0.5)
        return DetectionResult(score=score, label=label, confidence=conf, metadata={"path": p})


class AudioDetectorStub(BaseDetector):
    modality = "audio"

    def predict(self, sample: str | Path) -> DetectionResult:
        p = str(sample)
        score = 0.35 + 0.01 * (len(p) % 37)
        label: Label = "ai" if score >= self.threshold else "human"
        conf = 0.75
        return DetectionResult(score=score, label=label, confidence=conf, metadata={"path": p})


def _build_detector(name: str, cfg: DetectZooConfig, **kwargs: Any) -> BaseDetector:
    mod = modality_for(name)
    if mod == "text":
        det = TextDetectorStub(cfg, **kwargs)
    elif mod == "image":
        det = ImageDetectorStub(cfg, **kwargs)
    else:
        det = AudioDetectorStub(cfg, **kwargs)
    det.name = name
    return det


def load_detector(name: str, cfg: DetectZooConfig | None = None, **kwargs: Any) -> BaseDetector:
    """Factory matching DetectZoo load_detector (Section 3.1)."""
    cfg = cfg or DetectZooConfig()
    key = name.lower().strip().replace("-", "_")
    return _build_detector(key, cfg, **kwargs)
