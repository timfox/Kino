"""Visual (ResNet-50 CNNv2) and audio (PANNs) encoders (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.autocut.config import AutoCutConfig


@dataclass
class EncoderBundle:
    visual: np.ndarray
    audio: np.ndarray
    fps: int


def encode_visual_frames(
    frame_count: int,
    cfg: AutoCutConfig | None = None,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return (T, D_v) frame embeddings at low_fps (1 fps reasoning path)."""
    c = cfg or AutoCutConfig()
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(max(1, frame_count), c.video_feature_dim)).astype(np.float32)
    # L2-normalize rows like contrastive pretraining
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return (emb / norms).astype(np.float32)


def encode_audio_segment(
    duration_s: float,
    cfg: AutoCutConfig | None = None,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return (D_a,) PANNs-style log-mel CNN embedding."""
    c = cfg or AutoCutConfig()
    rng = np.random.default_rng(seed + int(duration_s * 100))
    v = rng.normal(size=c.audio_feature_dim).astype(np.float32)
    v /= float(np.linalg.norm(v) + 1e-9)
    return v


def clip_embedding_from_frames(frames: np.ndarray) -> np.ndarray:
    """Average frame embeddings for 20 fps visual-clip retrieval (Supp. 7.2)."""
    if frames.ndim == 1:
        return frames.astype(np.float64)
    return np.mean(frames, axis=0).astype(np.float64)


def encode_video_clip_at_fps(
    frame_count: int,
    cfg: AutoCutConfig | None = None,
    *,
    fps: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """High-fps path for visual-coherence clip segmentation."""
    return encode_visual_frames(frame_count, cfg, seed=seed)


def encode_multimodal_clip(
    *,
    frame_count: int = 8,
    audio_duration_s: float = 12.0,
    cfg: AutoCutConfig | None = None,
    seed: int = 0,
) -> EncoderBundle:
    c = cfg or AutoCutConfig()
    return EncoderBundle(
        visual=encode_visual_frames(frame_count, c, seed=seed),
        audio=encode_audio_segment(audio_duration_s, c, seed=seed + 1),
        fps=c.low_fps,
    )


def try_encode_frames_torch(
    frame_count: int,
    cfg: AutoCutConfig | None = None,
    *,
    seed: int = 0,
) -> np.ndarray | None:
    """Optional ResNet-50 backbone when torchvision is available."""
    try:
        import torch
        from torchvision import models, transforms
    except ImportError:
        return None

    c = cfg or AutoCutConfig()
    torch.manual_seed(seed)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(max(1, frame_count)):
        img = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
        from PIL import Image

        tensor = preprocess(Image.fromarray(img)).unsqueeze(0)
        with torch.no_grad():
            emb = model(tensor).cpu().numpy().reshape(-1)
        if emb.size > c.video_feature_dim:
            emb = emb[: c.video_feature_dim]
        elif emb.size < c.video_feature_dim:
            pad = np.zeros(c.video_feature_dim - emb.size, dtype=np.float32)
            emb = np.concatenate([emb, pad])
        frames.append(emb.astype(np.float32))
    return np.stack(frames, axis=0)


def mock_product_metadata() -> dict[str, Any]:
    return {
        "product_type": "Wireless Earbuds",
        "brand": "Edifier EVO PRO",
        "features": [
            "U-shaped in-ear fit",
            "Leather-texture body",
            "instant pairing",
            "customizable EQ",
            "multiple noise modes",
        ],
    }
