"""CineMatte-4K dataset loader (VP image subset + video metadata)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class CineMatte4KSample:
    """One VP matting example: scene frame, background plate, expert alpha."""

    image_path: Path
    background_path: Path
    alpha_path: Path
    split: str = "train"
    clip_id: str | None = None
    camera_trajectory: str | None = None  # path to trajectory JSON for video subset


def _resolve_path(root: Path, entry: dict, key: str) -> Path:
    p = Path(entry[key])
    return p if p.is_absolute() else (root / p)


def load_manifest(root: str | Path) -> list[CineMatte4KSample]:
    """
    Load ``cinematte4k_manifest.json`` from dataset root.

    Example manifest entry::

        {
          "image": "images/scene_0001.png",
          "background": "plates/bg_0001.png",
          "alpha": "alpha/0001.png",
          "split": "test"
        }
    """
    root_p = Path(root).expanduser().resolve()
    manifest_path = root_p / "cinematte4k_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. See documents/CINEMATTE.md for layout."
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: list[CineMatte4KSample] = []
    for row in data.get("samples", data if isinstance(data, list) else []):
        samples.append(
            CineMatte4KSample(
                image_path=_resolve_path(root_p, row, "image"),
                background_path=_resolve_path(root_p, row, "background"),
                alpha_path=_resolve_path(root_p, row, "alpha"),
                split=str(row.get("split", "train")),
                clip_id=row.get("clip_id"),
                camera_trajectory=row.get("camera_trajectory"),
            )
        )
    return samples


def write_manifest_template(path: str | Path, *, num_placeholders: int = 3) -> Path:
    """Write a starter manifest for local CineMatte-4K-style drops."""
    out = Path(path).expanduser().resolve()
    rows = [
        {
            "image": f"images/scene_{i:04d}.png",
            "background": f"plates/bg_{i:04d}.png",
            "alpha": f"alpha/{i:04d}.png",
            "split": "test" if i == 0 else "train",
        }
        for i in range(num_placeholders)
    ]
    payload = {
        "paper": "arXiv:2605.18328",
        "dataset": "CineMatte-4K-template",
        "notes": "Replace paths with VP stage captures (4K HDR, green-screen insertion workflow).",
        "samples": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


class CineMatte4KImageDataset(Dataset):
    """PyTorch dataset for CineMatte-4K image pairs."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = "test",
        long_edge: int = 1024,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.long_edge = long_edge
        all_samples = load_manifest(self.root)
        if split is not None:
            all_samples = [s for s in all_samples if s.split == split]
        if not all_samples:
            raise ValueError(f"No samples for split={split!r} under {self.root}")
        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        from ltx_trainer.cinematte.pipeline import _load_rgb

        s = self.samples[idx]
        device = torch.device("cpu")
        # Preserve aspect: scale by long edge
        from PIL import Image

        with Image.open(s.image_path) as im:
            w, h = im.size
        scale = self.long_edge / max(w, h)
        size = (int(w * scale), int(h * scale))
        image = _load_rgb(s.image_path, device, size)
        background = _load_rgb(s.background_path, device, size)
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        with Image.open(s.alpha_path) as im:
            im = im.convert("L")
            if size is not None:
                im = im.resize(size, Image.Resampling.BILINEAR)
            alpha = to_tensor(im).to(device).unsqueeze(0)
        return image, background, alpha.clamp(0, 1)


def iter_video_clips(root: str | Path) -> Iterator[dict]:
    """Yield video-subset metadata rows (tracked camera trajectories)."""
    root_p = Path(root).expanduser().resolve()
    clips_json = root_p / "video_clips.json"
    if not clips_json.is_file():
        return iter(())
    data = json.loads(clips_json.read_text(encoding="utf-8"))
    for clip in data.get("clips", []):
        yield clip
