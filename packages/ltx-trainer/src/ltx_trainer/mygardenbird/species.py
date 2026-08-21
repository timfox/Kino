"""Twelve MyGardenBird species catalog (Table 2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mygardenbird.config import MygardenbirdConfig

SPECIES: tuple[dict[str, Any], ...] = (
    {
        "common": "Asian Koel",
        "scientific": "Eudynamys scolopaceus",
        "ebird": "asikoe2",
        "clips_16k": 600,
        "clips_44k": 574,
        "xc_files": 136,
    },
    {
        "common": "Collared Kingfisher",
        "scientific": "Todiramphus chloris",
        "ebird": "colkin1",
        "clips_16k": 600,
        "clips_44k": 571,
        "xc_files": 118,
    },
    {
        "common": "Common Iora",
        "scientific": "Aegithina tiphia",
        "ebird": "comior1",
        "clips_16k": 600,
        "clips_44k": 598,
        "xc_files": 105,
    },
    {
        "common": "Common Tailorbird",
        "scientific": "Orthotomus sutorius",
        "ebird": "comtai1",
        "clips_16k": 600,
        "clips_44k": 577,
        "xc_files": 96,
    },
    {
        "common": "Coppersmith Barbet",
        "scientific": "Psilopogon haemacephalus",
        "ebird": "copbar1",
        "clips_16k": 600,
        "clips_44k": 574,
        "xc_files": 120,
    },
    {
        "common": "Large-tailed Nightjar",
        "scientific": "Caprimulgus macrurus",
        "ebird": "latnig2",
        "clips_16k": 600,
        "clips_44k": 579,
        "xc_files": 108,
    },
    {
        "common": "Olive-backed Sunbird",
        "scientific": "Cinnyris jugularis",
        "ebird": "olbsun4",
        "clips_16k": 600,
        "clips_44k": 578,
        "xc_files": 105,
    },
    {
        "common": "Spotted Dove",
        "scientific": "Spilopelia chinensis",
        "ebird": "spodov",
        "clips_16k": 600,
        "clips_44k": 599,
        "xc_files": 97,
    },
    {
        "common": "White-breasted Waterhen",
        "scientific": "Amaurornis phoenicurus",
        "ebird": "whbwat1",
        "clips_16k": 600,
        "clips_44k": 572,
        "xc_files": 115,
    },
    {
        "common": "White-throated Kingfisher",
        "scientific": "Halcyon smyrnensis",
        "ebird": "whtkin2",
        "clips_16k": 600,
        "clips_44k": 570,
        "xc_files": 123,
    },
    {
        "common": "Yellow-vented Bulbul",
        "scientific": "Pycnonotus goiavier",
        "ebird": "yevbul1",
        "clips_16k": 600,
        "clips_44k": 581,
        "xc_files": 118,
    },
    {
        "common": "Pied Fantail",
        "scientific": "Rhipidura javanica",
        "ebird": "piefan1",
        "clips_16k": 600,
        "clips_44k": 577,
        "xc_files": 140,
    },
)


def species_table() -> list[dict[str, Any]]:
    return [dict(s) for s in SPECIES]


def clip_id(source_id: str, onset_ms: int) -> str:
    return f"xc{source_id}_{onset_ms}"


def verify_balance(cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MygardenbirdConfig()
    totals_16k = sum(s["clips_16k"] for s in SPECIES)
    totals_xc = sum(s["xc_files"] for s in SPECIES)
    per_class_ok = all(s["clips_16k"] == cfg.clips_per_species for s in SPECIES)
    return {
        "n_species": len(SPECIES),
        "clips_16k_total": totals_16k,
        "xc_files_total": totals_xc,
        "balanced_600_per_class": per_class_ok,
        "matches_config": totals_16k == cfg.clips_16k and totals_xc == cfg.source_recordings,
    }


def species_demo() -> dict[str, Any]:
    bal = verify_balance()
    return {
        "species": [s["ebird"] for s in SPECIES],
        "balanced": bal["balanced_600_per_class"],
        "example_file_id": clip_id("1002657", 2860),
    }
