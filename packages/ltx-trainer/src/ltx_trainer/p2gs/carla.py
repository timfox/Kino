"""CARLA exposure-variation dataset helpers (P2GS Appendix A)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CarlaExposureSplit:
    """Metadata for ISO-Const vs ISO-Var training splits."""

    name: str
    iso_mode: str
    iso_std: float | None
    scene_dir: Path
    cameras_json: Path

    @property
    def is_ground_truth_uniform(self) -> bool:
        return self.iso_mode == "const"


def load_carla_manifest(root: str | Path) -> list[CarlaExposureSplit]:
    """
    Load ``carla_manifest.json`` if present; otherwise infer subdirs.

    Expected layout::

        root/
          iso_const_town01/
            cameras.json
            images/
          iso_std2_town01/
            ...
    """
    root_p = Path(root).expanduser().resolve()
    manifest = root_p / "carla_manifest.json"
    splits: list[CarlaExposureSplit] = []
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in data.get("splits", []):
            scene = root_p / entry["scene"]
            splits.append(
                CarlaExposureSplit(
                    name=entry.get("name", entry["scene"]),
                    iso_mode=entry.get("iso_mode", "var"),
                    iso_std=entry.get("iso_std"),
                    scene_dir=scene,
                    cameras_json=scene / "cameras.json",
                )
            )
        return splits

    for sub in sorted(root_p.iterdir()):
        if not sub.is_dir():
            continue
        cj = sub / "cameras.json"
        if not cj.is_file():
            continue
        name = sub.name.lower()
        if "const" in name or "iso8" in name:
            mode, std = "const", None
        elif "std2" in name:
            mode, std = "var", 2.0
        elif "std4" in name:
            mode, std = "var", 4.0
        else:
            mode, std = "var", None
        splits.append(
            CarlaExposureSplit(
                name=sub.name,
                iso_mode=mode,
                iso_std=std,
                scene_dir=sub,
                cameras_json=cj,
            )
        )
    return splits


def carla_training_plan() -> dict[str, Any]:
    """Recommended P2GS training/eval protocol from paper Sec. 4.4."""
    return {
        "paper": "arXiv:2605.16925",
        "train": ["iso_std2", "iso_std4"],
        "eval_gt": "iso_const (ISO 8)",
        "metrics": ["PSNR", "SSIM", "LPIPS", "HIS", "Std-Luminance", "delta_PSNR"],
        "notes": "Train on noisy ISO; evaluate reconstruction against uniform-illumination holdout.",
    }
