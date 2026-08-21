"""Client-side MOC engine stub (WASM Rust MOC analogue, Sec. 4.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ltx_trainer.survey_footprint.config import FULL_SKY_DEG2
from ltx_trainer.survey_footprint.surveys import SURVEY_BY_ID, SurveyRecord


@dataclass
class SphericalPatch:
    """Equirectangular patch (degrees); RA may wrap."""

    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float

    def contains(self, ra: float, dec: float) -> bool:
        if dec < self.dec_min or dec > self.dec_max:
            return False
        ra_n = ra % 360.0
        lo = self.ra_min % 360.0
        hi = self.ra_max % 360.0
        if lo <= hi:
            return lo <= ra_n <= hi
        return ra_n >= lo or ra_n <= hi


@dataclass
class MocFootprint:
    """Union of patches + target sky fraction from Table 1."""

    survey_id: str
    patches: list[SphericalPatch] = field(default_factory=list)
    sky_fraction: float = 0.0

    def contains(self, ra: float, dec: float) -> bool:
        if self.patches:
            return any(p.contains(ra, dec) for p in self.patches)
        return _hash_fraction(self.survey_id, ra, dec) < self.sky_fraction


def _hash_fraction(survey_id: str, ra: float, dec: float) -> float:
    """Deterministic [0,1) pseudo-coverage for patch-free surveys."""
    key = (survey_id, int(ra * 1000) % 360_000, int(dec * 1000) % 180_000)
    return (hash(key) % 10_000) / 10_000.0


def _default_patches(rec: SurveyRecord) -> list[SphericalPatch]:
    """Approximate footprints for overlap demos (not survey-definitive)."""
    sid = rec.survey_id
    if sid == "kids":
        return [SphericalPatch(140, 200, -35, -20)]
    if sid in ("euclid_dr1", "lsst_wfd", "roman_hlwas"):
        # Shared high-latitude extragalactic band (Use Case A)
        return [SphericalPatch(120, 240, -5, 60)]
    if sid == "erass1":
        return [SphericalPatch(0, 360, -90, 90)]
    if sid == "roman_hltds_deep":
        return [SphericalPatch(150, 152, 1, 3), SphericalPatch(35, 37, -5, -3)]
    return []


def load_builtin_moc(survey_id: str) -> MocFootprint:
    rec = SURVEY_BY_ID[survey_id]
    patches = _default_patches(rec)
    return MocFootprint(survey_id=survey_id, patches=patches, sky_fraction=rec.sky_fraction)


def load_custom_moc(survey_id: str, patches: list[SphericalPatch] | None = None) -> MocFootprint:
    return MocFootprint(survey_id=survey_id, patches=patches or [], sky_fraction=0.01)


def intersect_mocs(mocs: list[MocFootprint]) -> MocFootprint:
    """Set intersection (patch-wise AND + hash fallback)."""
    if not mocs:
        return MocFootprint("empty", sky_fraction=0.0)
    sid = "+".join(m.survey_id for m in mocs)
    return MocFootprint(survey_id=sid, patches=[], sky_fraction=min(m.sky_fraction for m in mocs))


def intersection_area_deg2(
    survey_ids: list[str],
    *,
    samples: int = 20_000,
    seed: int = 42,
) -> float:
    """Monte Carlo intersection area on the sphere (Sec. 4.2)."""
    if not survey_ids:
        return 0.0
    mocs = [load_builtin_moc(s) for s in survey_ids]
    rng = _lcg(seed)
    hits = 0
    for _ in range(samples):
        ra, dec = _random_angles(rng)
        if all(m.contains(ra, dec) for m in mocs):
            hits += 1
    return FULL_SKY_DEG2 * hits / samples


def union_area_deg2(survey_ids: list[str], *, samples: int = 20_000, seed: int = 42) -> float:
    if not survey_ids:
        return 0.0
    mocs = [load_builtin_moc(s) for s in survey_ids]
    rng = _lcg(seed)
    hits = 0
    for _ in range(samples):
        ra, dec = _random_angles(rng)
        if any(m.contains(ra, dec) for m in mocs):
            hits += 1
    return FULL_SKY_DEG2 * hits / samples


def filter_coordinates(
    ra_deg: list[float],
    dec_deg: list[float],
    moc: MocFootprint,
) -> list[bool]:
    """filterCoos() membership test (Sec. 2.5)."""
    return [moc.contains(ra, dec) for ra, dec in zip(ra_deg, dec_deg, strict=True)]


def _lcg(seed: int):
    state = seed

    def step() -> float:
        nonlocal state
        state = (1_103_515_245 * state + 12_345) % (2**31)
        return state / (2**31)

    return step


def _random_angles(rng) -> tuple[float, float]:
    u = rng()
    v = rng()
    dec = math.degrees(math.asin(2.0 * v - 1.0))
    ra = 360.0 * u
    return ra, dec
