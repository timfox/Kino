"""USDA 1.0 emit/parse for Love Dynasty / CID shot graphs (stdlib only)."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.freeusd.config import DEFAULT_TWO_SHOT, FORMAT, FreeUSDConfig, SET_PRIMS, SpatialPose


def _q(s: str) -> str:
    t = (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{t}"'


def _prim(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", (name or "unnamed").strip())
    s = s.strip("_") or "unnamed"
    if s[0].isdigit():
        s = "p_" + s
    return s[:48]


def parse_blocking(beat: str, *, default: tuple[SpatialPose, ...] = DEFAULT_TWO_SHOT) -> list[SpatialPose]:
    """Rule spatial-LLM: map soap beat language onto left/right marks."""
    text = (beat or "").lower()
    elena, julian = default[0], default[1]
    if "doorway" in text or "stands" in text or "stand" in text:
        julian = SpatialPose(julian.name, julian.prim, julian.translate, "standing", "doorway from beat")
    if "sits" in text or "sit" in text or "sofa" in text:
        elena = SpatialPose(elena.name, elena.prim, elena.translate, "sitting", "sofa from beat")
    return [elena, julian]


def shot_to_usda(
    shot: dict[str, Any],
    *,
    cfg: FreeUSDConfig | None = None,
    poses: list[SpatialPose] | None = None,
) -> str:
    """Compose one TV shot as a readable USDA layer for an LLM / CID."""
    cfg = cfg or FreeUSDConfig()
    slate = str(shot.get("slate_label") or shot.get("slate") or "S001E001 Scene 1 Shot 1")
    stem = str(shot.get("tv_stem") or "S001E001_SC01_SH01")
    heading = str(shot.get("scene_heading") or "INT. VALE LIVING ROOM - DAY")
    beat = str((shot.get("blocking") or {}).get("action") or shot.get("beat") or "")
    poses = poses if poses is not None else parse_blocking(beat)
    root = _prim(stem)
    cont = shot.get("continuity") or {}
    fps = float(shot.get("fps") or cont.get("fps") or cfg.fps)
    frames = int(shot.get("duration_frames") or cont.get("frames") or cfg.frames_default)
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{root}"',
        f'    doc = """{FORMAT} FreeUSD shot graph — spatial lock for CID compose."""',
        f"    metersPerUnit = {cfg.meters_per_unit}",
        f'    upAxis = "{cfg.up_axis}"',
        f"    framesPerSecond = {fps}",
        f"    timeCodesPerSecond = {fps}",
        "    startTimeCode = 0",
        f"    endTimeCode = {max(frames - 1, 0)}",
        ")",
        "",
        f'def Xform "{root}" (',
        '    kind = "assembly"',
        ")",
        "{",
        f"    custom uniform string gopex_usd_format = {_q(FORMAT)}",
        f"    custom uniform string gopex_slate = {_q(slate)}",
        f"    custom uniform string gopex_tv_stem = {_q(stem)}",
        f"    custom uniform string gopex_scene_heading = {_q(heading)}",
        f"    custom uniform string gopex_shot_size = {_q(str(shot.get('shot_size') or cfg.shot_size))}",
        f"    custom uniform string gopex_lighting = {_q(cfg.lighting)}",
        f"    custom uniform int gopex_expected_cast = {cfg.expected_cast}",
        f"    custom uniform string gopex_beat = {_q(beat)}",
        "",
        '    def Camera "Cam_A" (',
        '        kind = "component"',
        "    )",
        "    {",
        "        double3 xformOp:translate = (0, 1.55, 0)",
        '        uniform token[] xformOpOrder = ["xformOp:translate"]',
        f"        custom uniform float gopex_lens_mm = {cfg.camera_lens_mm}",
        '        custom uniform string gopex_move = "locked-off"',
        '        custom uniform string gopex_angle = "eye-level medium two-shot"',
        "    }",
        "",
    ]
    for pose in poses:
        px, py, pz = pose.translate
        lines.extend(
            [
                f'    def Xform "{_prim(pose.prim)}" (',
                '        kind = "component"',
                "    )",
                "    {",
                f"        double3 xformOp:translate = ({px}, {py}, {pz})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                f"        custom uniform string gopex_character_name = {_q(pose.name)}",
                f"        custom uniform string gopex_pose = {_q(pose.pose)}",
                f"        custom uniform string gopex_notes = {_q(pose.notes)}",
                "    }",
                "",
            ]
        )
    for prim in SET_PRIMS:
        px, py, pz = prim.translate
        lines.extend(
            [
                f'    def Xform "{_prim(prim.prim)}" (',
                '        kind = "component"',
                "    )",
                "    {",
                f"        double3 xformOp:translate = ({px}, {py}, {pz})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                f"        custom uniform string gopex_set_piece = {_q(prim.name)}",
                f"        custom uniform string gopex_notes = {_q(prim.notes)}",
                "    }",
                "",
            ]
        )
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def parse_expected_cast(usda: str) -> int:
    m = re.search(r"gopex_expected_cast\s*=\s*(\d+)", usda)
    return int(m.group(1)) if m else 2


def parse_lighting(usda: str) -> str:
    m = re.search(r'gopex_lighting\s*=\s*"([^"]*)"', usda)
    return (m.group(1) if m else "high-key").strip()


def parse_character_names(usda: str) -> list[str]:
    return re.findall(r'gopex_character_name\s*=\s*"([^"]+)"', usda)


def usda_to_spatial_lock(usda: str) -> str:
    """Linearize USDA into an LTX prompt prefix a spatial LLM would emit."""
    names = parse_character_names(usda)
    lighting = parse_lighting(usda)
    n = parse_expected_cast(usda)
    poses = re.findall(
        r'gopex_character_name\s*=\s*"([^"]+)"[\s\S]{0,240}?gopex_pose\s*=\s*"([^"]+)"[\s\S]{0,120}?gopex_notes\s*=\s*"([^"]*)"',
        usda,
    )
    bits = [f"Spatial lock (FreeUSD): exactly {n} people"]
    if poses:
        for name, pose, notes in poses:
            bits.append(f"{name} {pose} ({notes})" if notes else f"{name} {pose}")
    elif names:
        bits.append(" and ".join(names))
    bits.append(f"{lighting} three-point studio lighting, 35mm locked-off MS two-shot")
    bits.append("no extras, no crowd, both faces sharp")
    return "; ".join(bits) + "."
