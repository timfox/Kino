"""Photographic Scene Graph (Sec. 3.4, Eq. 4–5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.before_shutter.config import BeforeShutterConfig


@dataclass
class SceneNode:
    name: str
    kind: str  # "non_emissive" | "emissive"
    affordance: str | None = None
    ev100: float | None = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SpatialEdge:
    src: str
    dst: str
    relation: str  # left-of, sit-on, stand-on, lit, ambient, ...


@dataclass
class EmissiveEdge:
    emitter: str
    target: str
    influence: float = 1.0


@dataclass
class EmitterRatio:
    emitter_a: str
    emitter_b: str
    ratio: float
    distance_a_m: float
    distance_b_m: float


@dataclass
class PhotographicSceneGraph:
    """G = (V_non, V_emi, E_n2n, E_e2n, E_e2e) — paper Eq. (4)."""

    nodes_non: dict[str, SceneNode] = field(default_factory=dict)
    nodes_emi: dict[str, SceneNode] = field(default_factory=dict)
    spatial: list[SpatialEdge] = field(default_factory=list)
    emissive_to_non: list[EmissiveEdge] = field(default_factory=list)
    emitter_ratios: list[EmitterRatio] = field(default_factory=list)

    def add_non(self, name: str, *, affordance: str | None = None, ev100: float | None = None) -> None:
        self.nodes_non[name] = SceneNode(name, "non_emissive", affordance=affordance, ev100=ev100)

    def add_emi(self, name: str, *, ev100: float | None = None) -> None:
        self.nodes_emi[name] = SceneNode(name, "emissive", ev100=ev100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "V_non": list(self.nodes_non.keys()),
            "V_emi": list(self.nodes_emi.keys()),
            "E_n2n": [(e.src, e.relation, e.dst) for e in self.spatial],
            "E_e2n": [(e.emitter, e.target, e.influence) for e in self.emissive_to_non],
            "E_e2e": [
                (r.emitter_a, r.emitter_b, round(r.ratio, 3)) for r in self.emitter_ratios
            ],
        }


def probe_emitter_ratios(
    meter_amb: float,
    meter_with_emitter: dict[str, float],
    distances_m: dict[str, float],
    *,
    cfg: BeforeShutterConfig | None = None,
) -> list[EmitterRatio]:
    """Ambient subtraction ratios r_A:amb and r_A:B (Eq. 5)."""
    _ = cfg
    amb = max(meter_amb, 1e-6)
    out: list[EmitterRatio] = []
    deltas = {k: max(v - amb, 0.0) for k, v in meter_with_emitter.items()}
    for name, delta in deltas.items():
        r_amb = delta / amb
        out.append(
            EmitterRatio(
                emitter_a=name,
                emitter_b="ambient",
                ratio=r_amb,
                distance_a_m=distances_m.get(name, 1.0),
                distance_b_m=1.0,
            )
        )
    keys = list(deltas.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            da = max(distances_m.get(a, 1.0), 1e-3)
            db = max(distances_m.get(b, 1.0), 1e-3)
            ratio = (deltas[a] / max(deltas[b], 1e-6)) * ((da / db) ** 2)
            out.append(
                EmitterRatio(
                    emitter_a=a,
                    emitter_b=b,
                    ratio=ratio,
                    distance_a_m=da,
                    distance_b_m=db,
                )
            )
    return out


def build_demo_graph(*, prompt: str = "Melancholy") -> PhotographicSceneGraph:
    """Minimal indoor graph aligned with Fig. 6 (composition + lighting)."""
    g = PhotographicSceneGraph()
    g.add_non("face", ev100=-0.2)
    g.add_non("torso", ev100=-2.0)
    g.add_non("hand", ev100=-3.2)
    g.add_non("red_chair", affordance="sit-on", ev100=-3.9)
    g.add_non("window", ev100=0.8)
    g.add_non("wall_light", ev100=1.0)
    g.add_non("floor", affordance="stand-on")
    g.add_emi("ambient")
    g.add_emi("key")
    g.add_emi("fill")
    g.spatial.extend(
        [
            SpatialEdge("hand", "red_chair", "left-of"),
            SpatialEdge("torso", "red_chair", "sit-on"),
            SpatialEdge("face", "window", "lit"),
            SpatialEdge("torso", "floor", "stand-on"),
        ]
    )
    g.emissive_to_non.extend(
        [
            EmissiveEdge("key", "face", 0.43),
            EmissiveEdge("fill", "face", 0.03),
            EmissiveEdge("ambient", "torso", 0.07),
        ]
    )
    g.emitter_ratios = probe_emitter_ratios(
        meter_amb=0.07,
        meter_with_emitter={"key": 0.50, "fill": 0.10},
        distances_m={"key": 1.0, "fill": 1.5},
    )
    g.nodes_non["face"].ev100 = -0.2 if "Melancholy" in prompt else 0.2
    return g
