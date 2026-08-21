"""RLE dataset manifest loader (1024×768 VP beam-splitter captures, Sec. 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ltx_trainer.eic_lie.events import events_to_sbt_voxel


@dataclass
class RLESample:
    low_path: Path
    normal_path: Path
    events_path: Path | None
    split: str = "test"


def load_rle_manifest(root: str | Path) -> list[RLESample]:
    root_p = Path(root).expanduser().resolve()
    manifest = root_p / "rle_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}; see documents/EIC_LIE.md")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = data.get("samples", data if isinstance(data, list) else [])
    out: list[RLESample] = []
    for row in rows:
        def _p(key: str) -> Path:
            p = Path(row[key])
            return p if p.is_absolute() else root_p / p

        out.append(
            RLESample(
                low_path=_p("low"),
                normal_path=_p("normal"),
                events_path=_p("events") if row.get("events") else None,
                split=str(row.get("split", "test")),
            )
        )
    return out


def write_rle_manifest_template(path: str | Path, *, n: int = 2) -> Path:
    out = Path(path).expanduser().resolve()
    samples = [
        {
            "low": f"low/{i:04d}.png",
            "normal": f"normal/{i:04d}.png",
            "events": f"events/{i:04d}.npy",
            "split": "test" if i == 0 else "train",
        }
        for i in range(n)
    ]
    payload = {
        "paper": "arXiv:2605.22186",
        "dataset": "RLE-template",
        "resolution": "1024x768",
        "samples": samples,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


class RLEDataset(Dataset):
    def __init__(self, root: str | Path, *, split: str = "test", event_bins: int = 5) -> None:
        self.root = Path(root)
        self.event_bins = event_bins
        self.samples = [s for s in load_rle_manifest(root) if s.split == split]
        if not self.samples:
            raise ValueError(f"No RLE samples for split={split!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        from ltx_trainer.eic_lie.pipeline import _load_rgb

        s = self.samples[idx]
        low = _load_rgb(s.low_path, torch.device("cpu"))
        normal = _load_rgb(s.normal_path, torch.device("cpu"))
        if s.events_path and s.events_path.suffix == ".npy":
            import numpy as np

            arr = np.load(s.events_path)
            # Expect [N,4] x,y,t,p
            events = [(int(r[0]), int(r[1]), float(r[2]), int(r[3])) for r in arr]
            h, w = low.shape[-2:]
            voxel = events_to_sbt_voxel(events, height=h, width=w, num_bins=self.event_bins)
        else:
            from ltx_trainer.eic_lie.events import synthetic_events_from_image

            ev = synthetic_events_from_image(normal)
            h, w = low.shape[-2:]
            voxel = events_to_sbt_voxel(ev, height=h, width=w, num_bins=self.event_bins)
        return low, normal, voxel
