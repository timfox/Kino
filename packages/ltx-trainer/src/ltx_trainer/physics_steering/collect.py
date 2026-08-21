"""Collect VideoMAE pooled activations from IntPhys videos (Sec. 4.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from ltx_trainer.physics_steering.activations_io import save_layer_activations
from ltx_trainer.physics_steering.config import PhysicsSteeringConfig
from ltx_trainer.physics_steering.videomae_hooks import VideoMAEActivationCollector


def load_video_tensor(
    path: Path,
    *,
    num_frames: int,
    spatial_size: int,
) -> Tensor:
    """``(1, T, C, H, W)`` float tensor in [0, 1] for VideoMAE."""
    from ltx_trainer.video_utils import read_video

    frames, _ = read_video(str(path), max_frames=num_frames)
    if frames.shape[0] < 1:
        raise ValueError(f"no frames in {path}")
    # frames: (T, H, W, C) uint8
    t = frames.shape[0]
    if t < num_frames:
        pad = frames[-1:].repeat(num_frames - t, 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)
    elif t > num_frames:
        idx = torch.linspace(0, t - 1, num_frames).long()
        frames = frames[idx]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    frames = torch.nn.functional.interpolate(
        frames,
        size=(spatial_size, spatial_size),
        mode="bilinear",
        align_corners=False,
    )
    return frames.unsqueeze(0)


def forward_videomae_pooled(
    pixel_values: Tensor,
    model: Any,
    processor: Any,
    *,
    num_layers: int,
) -> dict[int, Tensor]:
    """Run VideoMAE and return per-layer mean-pooled ``(D,)`` vectors."""
    collector = VideoMAEActivationCollector(model, num_layers=num_layers)
    collector.register()
    collector.clear()
    device = next(model.parameters()).device
    inputs = processor(list(pixel_values[0].permute(0, 2, 3, 1).cpu().numpy()), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)
    pooled = collector.pooled_by_layer()
    return {layer: vec[0] if vec.dim() > 1 else vec for layer, vec in pooled.items()}


def merge_layer_dicts(
    acc: dict[int, list[Tensor]],
    sample: dict[int, Tensor],
) -> None:
    for layer, vec in sample.items():
        acc.setdefault(layer, []).append(vec.cpu())


def stack_layer_features(acc: dict[int, list[Tensor]]) -> dict[int, Tensor]:
    return {layer: torch.stack(vecs, dim=0) for layer, vecs in acc.items()}


def collect_intphys_activations(
    records: list[Any],
    output: str | Path,
    *,
    cfg: PhysicsSteeringConfig | None = None,
    device: str | None = None,
    max_videos: int | None = None,
    progress: Callable[[int, int, Path], None] | None = None,
) -> dict[str, Any]:
    """Encode IntPhys videos; write NPZ for :func:`run_full_experiment`."""
    cfg = cfg or PhysicsSteeringConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    from ltx_trainer.physics_steering.videomae_hooks import load_videomae_model

    model, processor = load_videomae_model(cfg)
    model = model.to(device).eval()

    subset = records[:max_videos] if max_videos else records
    layer_acc: dict[int, list[Tensor]] = {}
    labels: list[int] = []
    blocks: list[int] = []
    skipped: list[str] = []

    for i, rec in enumerate(subset):
        path = Path(rec.path if hasattr(rec, "path") else rec["path"])
        if progress:
            progress(i + 1, len(subset), path)
        if not path.is_file():
            skipped.append(str(path))
            continue
        try:
            pixels = load_video_tensor(
                path,
                num_frames=cfg.num_frames,
                spatial_size=cfg.spatial_size,
            )
            pooled = forward_videomae_pooled(pixels, model, processor, num_layers=cfg.num_layers)
            merge_layer_dicts(layer_acc, pooled)
            labels.append(int(rec.label if hasattr(rec, "label") else rec["label"]))
            blocks.append(int(rec.block_id if hasattr(rec, "block_id") else rec["block_id"]))
        except Exception as exc:
            skipped.append(f"{path}: {exc}")

    if not labels:
        raise RuntimeError("no videos encoded; check GOPEX_INTPHYS_ROOT and ffmpeg/pyav")

    layer_features = stack_layer_features(layer_acc)
    label_t = torch.tensor(labels, dtype=torch.long)
    block_t = torch.tensor(blocks, dtype=torch.long)
    out_path = save_layer_activations(
        output,
        layer_features,
        label_t,
        block_t,
        metadata={
            "model_id": cfg.model_id,
            "n_frames": cfg.num_frames,
            "spatial_size": cfg.spatial_size,
            "source": "intphys",
        },
    )
    return {
        "saved": str(out_path),
        "n_encoded": len(labels),
        "n_skipped": len(skipped),
        "skipped_sample": skipped[:5],
        "layers": sorted(layer_features.keys()),
        "feature_shape": {str(k): list(v.shape) for k, v in layer_features.items()},
    }
