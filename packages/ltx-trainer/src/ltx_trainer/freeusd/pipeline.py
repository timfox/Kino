"""Framework card + CPU smoke for FreeUSD spatial compose."""

from __future__ import annotations

from typing import Any

from ltx_trainer.freeusd.compose import compose_shot
from ltx_trainer.freeusd.config import ARXIV, FORMAT, PAPER, FreeUSDConfig
from ltx_trainer.freeusd.score import score_frame_vs_usd
from ltx_trainer.freeusd.usda import usda_to_spatial_lock


def framework_card(cfg: FreeUSDConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FreeUSDConfig()
    return {
        "paper": PAPER,
        "arxiv": ARXIV,
        "method": {
            "format": cfg.format,
            "role": "OpenUSD ASCII scene graph as the spatial memory for TV shots",
            "compose": "rule spatial-LLM now; CID-trained text LLM later",
            "score": "CPU frame fail vs USDA expected cast/lighting/blocking",
            "cid": "lovedynasty-freeusd pipeline (CPU, no GPU runner)",
        },
        "config": {
            "format": cfg.format,
            "fps": cfg.fps,
            "expected_cast": cfg.expected_cast,
            "lighting": cfg.lighting,
            "shot_size": cfg.shot_size,
        },
    }


def _demo_shot() -> dict[str, Any]:
    return {
        "slate_label": "S001E001 Scene 1 Shot 1",
        "tv_stem": "S001E001_SC01_SH01",
        "scene_heading": "INT. VALE LIVING ROOM - DAY",
        "shot_size": "MS two-shot",
        "fps": 24,
        "duration_frames": 121,
        "blocking": {
            "action": "Living room cold open: she sits he stands in the doorway",
        },
        "prompt": "Elena Vale and Julian Hart, daytime soap, high-key living room.",
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    _ = seed
    shot = _demo_shot()
    composed = compose_shot(shot)
    lock = usda_to_spatial_lock(composed["usda"])
    noir_fail = score_frame_vs_usd(
        {
            "look": "noir",
            "look_ok": False,
            "person_count": 1,
            "person_count_ok": True,
            "shot_size_guess": "MCU",
            "expected_shot_size": "MS two-shot",
        },
        composed["usda"],
    )
    soap_pass = score_frame_vs_usd(
        {
            "look": "daytime",
            "look_ok": True,
            "person_count": 2,
            "person_count_ok": True,
            "shot_size_guess": "MS",
            "expected_shot_size": "MS two-shot",
        },
        composed["usda"],
    )
    return {
        "slate": shot["slate_label"],
        "spatial_lock": lock,
        "usda_has_camera": 'def Camera "Cam_A"' in composed["usda"],
        "usda_has_elena": "Char_Elena" in composed["usda"],
        "noir_failures": noir_fail["failures"],
        "soap_ok": soap_pass["soap_ok"],
        "format": FORMAT,
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    card = framework_card()
    return {
        "package": "freeusd",
        "paper": PAPER,
        "arxiv": ARXIV,
        "format": FORMAT,
        "usda_has_camera": demo["usda_has_camera"],
        "soap_ok": demo["soap_ok"],
        "noir_has_look_bleed": any("look_bleed" in f for f in demo["noir_failures"]),
        "cid_pipeline": "lovedynasty-freeusd",
        "method": card["method"]["compose"],
    }
