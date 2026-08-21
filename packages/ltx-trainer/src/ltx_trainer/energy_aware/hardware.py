"""Reference hardware tables from survey (Tables 2–4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExascaleSite:
    name: str
    region: str
    sustained_eflops: float
    power_mw: float
    usd_per_kwh: float
    kg_co2_per_kwh: float


EXASCALE_TOP500_NOV2025: tuple[ExascaleSite, ...] = (
    ExascaleSite("EL CAPITAN", "USA (California)", 1.809, 29.68, 0.148, 0.220),
    ExascaleSite("FRONTIER", "USA (Tennessee)", 1.353, 22.78, 0.148, 0.360),
    ExascaleSite("AURORA", "USA (Illinois)", 1.012, 38.69, 0.148, 0.320),
    ExascaleSite("JUPITER", "Germany (Jülich)", 1.000, 15.87, 0.285, 0.380),
    ExascaleSite("LUMI", "Finland", 0.0, 7.10, 0.285, 0.050),
)


@dataclass(frozen=True)
class AIAccelerator:
    name: str
    tdp_w: float
    peak_precision: str
    memory: str
    energy_note: str


AI_ACCELERATORS_2026: tuple[AIAccelerator, ...] = (
    AIAccelerator("NVIDIA B300 Ultra", 1400, "FP4/FP6/INT8", "288GB HBM3e", "liquid cooling standard"),
    AIAccelerator("AMD MI400X", 1600, "FP4/FP8/FP16", "432GB HBM4", "multi-die packaging"),
    AIAccelerator("Google TPU v7", 157, "BF16/FP8/INT8", "192GB HBM3e", "systolic array efficiency"),
    AIAccelerator("Intel Gaudi 3", 900, "BF16/FP8", "128GB HBM2e", "integrated RoCE"),
)


def exascale_table_dict() -> list[dict[str, Any]]:
    rows = []
    for s in EXASCALE_TOP500_NOV2025:
        rows.append(
            {
                "machine": s.name,
                "region": s.region,
                "power_mw": s.power_mw,
                "usd_per_hour": round(s.power_mw * 1000 * s.usd_per_kwh, 0),
                "kg_co2_per_hour": round(s.power_mw * 1000 * s.kg_co2_per_kwh, 0),
            }
        )
    return rows
