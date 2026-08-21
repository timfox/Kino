"""Qualitative benchmark cases from paper Fig. 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdCase:
    case_id: str
    product_type: str
    brand: str
    features: tuple[str, ...]
    script_lines: tuple[str, ...]
    clip_timestamps: tuple[str, ...]


BENCHMARK_CASES: tuple[AdCase, ...] = (
    AdCase(
        case_id="edifier_earbuds",
        product_type="Wireless Earbuds",
        brand="Edifier EVO PRO",
        features=(
            "U-shaped in-ear fit",
            "Leather-texture body",
            "instant pairing",
            "customizable EQ",
            "multiple noise modes",
        ),
        script_lines=(
            "Edifier EVO PRO, wireless earbuds, designed with a U-shaped in-ear fit,",
            "these earbuds connect instantly, offer customizable EQ, multiple noise modes,",
            "and plenty of ear tips for the perfect fit, a pair truly made for you!",
            "and a leather-textured case,",
        ),
        clip_timestamps=(
            "[Clip 1]: 1s ~ 2s",
            "[Clip 2]: 2s ~ 3s",
            "[Clip 3]: 3s ~ 5s",
            "[Clip 4]: 5s ~ 7s",
            "[Clip 5]: 7s ~ 8s",
        ),
    ),
    AdCase(
        case_id="bodoreme_face_cream",
        product_type="Children's face cream",
        brand="BODOREME",
        features=(
            "Pump-style dispenser",
            "Moisturizing but non-sticky",
            "Blue chamomile with natural plant lipids",
        ),
        script_lines=(
            "Moms, you really need to listen to me this time,",
            "you have to get your child's face cream ready in advance.",
            "This BODOREME children's face cream,",
            "Its cloud-soft texture spreads easily,",
            "It contains blue chamomile, combined with natural plant lipids,",
        ),
        clip_timestamps=(
            "[Clip 1]: 0s ~ 1s",
            "[Clip 2]: 1s ~ 3s",
            "[Clip 3]: 3s ~ 4s",
            "[Clip 4]: 4s ~ 5s",
            "[Clip 5]: 5s ~ 7s",
        ),
    ),
)


def case_by_id(case_id: str) -> AdCase | None:
    for c in BENCHMARK_CASES:
        if c.case_id == case_id:
            return c
    return None


def case_to_product_info(case: AdCase) -> dict[str, Any]:
    return {
        "product_type": case.product_type,
        "brand": case.brand,
        "features": list(case.features),
    }
