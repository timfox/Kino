"""Ray-decoupled visibility for enclosed scenes (Sec. 3.5, Eq. 5–6)."""

from __future__ import annotations

from ltx_trainer.lightharmony3d.config import RAY_TYPES_SHADOW, RAY_TYPES_TRANSPARENT


def ray_indicator(ray_type: str) -> float:
    """I(ω)=1 for shadow/diffuse, 0 for camera/transmission/glossy."""
    key = ray_type.lower()
    if key in RAY_TYPES_SHADOW:
        return 1.0
    if key in RAY_TYPES_TRANSPARENT:
        return 0.0
    return 0.5


def mix_bsdf(
    f_principled: float,
    f_transparent: float,
    *,
    ray_type: str,
) -> float:
    """Eq. (6): linear blend between opaque and transparent BSDF."""
    i = ray_indicator(ray_type)
    return i * f_principled + (1.0 - i) * f_transparent
