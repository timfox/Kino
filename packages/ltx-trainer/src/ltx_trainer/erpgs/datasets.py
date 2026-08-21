"""Public ERP NVS datasets referenced in ErpGS (Sec. 4.1)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "OmniBlender": {
            "source": "EgoNeRF / ODGS preprocessing",
            "type": "synthetic indoor/outdoor",
            "motion": "egocentric spiral",
        },
        "Ricoh360": {
            "source": "real outdoor Ricoh Theta",
            "type": "outdoor",
            "motion": "egocentric",
        },
        "OmniScenes": {
            "source": "PICCOLO / Kim et al. ICCV 2021",
            "type": "real indoor",
            "motion": "non-egocentric",
            "mask": "viewpoint-dependent obstacle mask used",
        },
    }
