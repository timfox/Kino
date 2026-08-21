"""Visual attribute vectors for numpy reference (rescaled to [0,1])."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.uaclip.demand import Platform


@dataclass
class VisualAttributes:
    colorfulness: float = 0.5
    brightness: float = 0.5
    symmetry: float = 0.5
    aesthetic: float = 0.5
    uniqueness: float = 0.5

    def to_dict(self, platform: Platform) -> dict[str, float]:
        d = {
            "colorfulness": self.colorfulness,
            "brightness": self.brightness,
            "symmetry": self.symmetry,
            "aesthetic": self.aesthetic,
            "uniqueness": self.uniqueness,
        }
        if platform == "airbnb":
            return {k: d[k] for k in ("uniqueness", "aesthetic")}
        return {k: d[k] for k in ("colorfulness", "brightness", "symmetry", "aesthetic")}

    @classmethod
    def from_vector(cls, v: np.ndarray, platform: Platform) -> VisualAttributes:
        v = np.clip(np.asarray(v, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if platform == "airbnb":
            if v.size < 2:
                v = np.pad(v, (0, max(0, 2 - v.size)), constant_values=0.5)
            return cls(uniqueness=float(v[0]), aesthetic=float(v[1]))
        if v.size < 4:
            v = np.pad(v, (0, max(0, 4 - v.size)), constant_values=0.5)
        return cls(
            colorfulness=float(v[0]),
            brightness=float(v[1]),
            symmetry=float(v[2]),
            aesthetic=float(v[3]),
        )

    @classmethod
    def random(cls, rng: np.random.Generator, platform: Platform) -> VisualAttributes:
        if platform == "amazon":
            return cls(
                colorfulness=float(rng.random()),
                brightness=float(rng.random()),
                symmetry=float(rng.random()),
                aesthetic=float(rng.random()),
            )
        return cls(
            uniqueness=float(rng.random()),
            aesthetic=float(rng.random()),
        )
