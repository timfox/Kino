"""Example forwardable work types from the paper (Sec. 5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimpleRay:
    """Minimal templated ray for smoke tests."""

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    pixel_id: int


@dataclass
class SchlierenFwdRay:
    """SchlieRaFI forwardable state (Listing 1)."""

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    tmin: float
    pixel_id: int
    integral: float
    surf_color: tuple[float, float, float]


@dataclass
class StreamlineParticle:
    """Particle advection work item (Sec. 5.4)."""

    particle_id: int
    position: tuple[float, float, float]


@dataclass
class NBodyParticle:
    """N-body migration particle (Listing 2)."""

    mass: float
    pos: tuple[float, float, float]
    vel: tuple[float, float, float]


@dataclass
class VirtualParticle:
    """Barnes–Hut essential-tree node (Listing 2)."""

    pos: tuple[float, float, float]
    mass: float
    smax: float
    source_rank: int


@dataclass
class RefinementReq:
    """Tree refinement request between ranks."""

    sender_rank: int
