"""USDA 1.0 emit/parse for Love Dynasty / CID shot graphs (stdlib only)."""

from __future__ import annotations

import re
import hashlib
import json
import os
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
    camera = shot.get("camera") or {}
    lighting = shot.get("lighting") or {}
    technical = shot.get("technical") or {}
    blocking = shot.get("blocking") or {}
    continuity = shot.get("continuity") or {}
    cast = shot.get("cast") or []
    # Keep one queryable JSON envelope in addition to the human-readable USD
    # attributes below. This makes the layer useful to CID, training audits,
    # rerender tools, and future DCC importers without inventing another sidecar.
    prompt = str(shot.get("prompt") or "")
    context = {
        "schema": "gopex.usda_context/v1",
        "production": {"series": shot.get("series"), "season": shot.get("season_number"), "episode": shot.get("episode_number"), "episode_title": shot.get("episode_title"), "production_code": shot.get("production_code"), "act": shot.get("act")},
        "scenario": {"scene_heading": heading, "scene_slug": shot.get("scene_slug"), "set": shot.get("set_name"), "time_of_day": shot.get("time_of_day"), "beat": beat, "blocking": blocking, "cast": cast, "sound": shot.get("sound") or {}, "continuity": continuity, "shot_context": shot.get("scenario") or {}},
        "camera": camera, "lighting": lighting,
        "training": {"product": technical.get("product"), "pipeline": technical.get("pipeline"), "hq": technical.get("hq"), "base": technical.get("base") or os.environ.get("GOPEX_LTX_MODEL_PATH") or os.environ.get("NATIVE_LTX"), "lora": technical.get("lora") or os.environ.get("LORA_CKPT"), "training_run": technical.get("training_run") or os.environ.get("GOPEX_PHASE2_RUN"), "checkpoint_step": technical.get("checkpoint_step") or os.environ.get("GOPEX_CHECKPOINT_STEP"), "sidecar_profile": os.environ.get("GOPEX_AV_FOLD_HOOKS", "")},
        "prompt": {"sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "text": prompt},
        "files": shot.get("files") or {},
        "evaluation": shot.get("evaluation") or {},
    }
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    evaluation = shot.get("evaluation") or {}
    scenario = shot.get("scenario") or {}
    scenario_prompt = str(scenario.get("source_prompt") or "")
    anchor = str(scenario.get("anchor_reference") or "")
    resolution = str(scenario.get("resolution") or "")
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
        f"    custom uniform string gopex_series = {_q(str(shot.get('series') or ''))}",
        f"    custom uniform string gopex_production_code = {_q(str(shot.get('production_code') or ''))}",
        f"    custom uniform string gopex_episode_title = {_q(str(shot.get('episode_title') or ''))}",
        f"    custom uniform int gopex_season_number = {int(shot.get('season_number') or 1)}",
        f"    custom uniform int gopex_episode_number = {int(shot.get('episode_number') or 1)}",
        f"    custom uniform int gopex_scene_number = {int(shot.get('scene_number') or 1)}",
        f"    custom uniform int gopex_shot_number = {int(shot.get('shot_number') or 1)}",
        f"    custom uniform int gopex_act = {int(shot.get('act') or 1)}",
        f"    custom uniform string gopex_set_name = {_q(str(shot.get('set_name') or ''))}",
        f"    custom uniform string gopex_time_of_day = {_q(str(shot.get('time_of_day') or ''))}",
        f"    custom uniform string gopex_shot_type = {_q(str(shot.get('shot_type') or ''))}",
        f"    custom uniform string gopex_camera_move = {_q(str(camera.get('move') or ''))}",
        f"    custom uniform string gopex_camera_focus = {_q(str(camera.get('focus') or ''))}",
        f"    custom uniform string gopex_prompt_sha256 = {_q(context['prompt']['sha256'])}",
        f"    custom uniform string gopex_training_base = {_q(str(context['training'].get('base') or ''))}",
        f"    custom uniform string gopex_training_lora = {_q(str(context['training'].get('lora') or ''))}",
        f"    custom uniform string gopex_training_run = {_q(str(context['training'].get('training_run') or ''))}",
        f"    custom uniform string gopex_sidecar_profile = {_q(str(context['training'].get('sidecar_profile') or ''))}",
        f"    custom uniform string gopex_scenario_prompt = {_q(scenario_prompt)}",
        f"    custom uniform string gopex_anchor_reference = {_q(anchor)}",
        f"    custom uniform string gopex_render_resolution = {_q(resolution)}",
        f"    custom uniform int gopex_render_frames = {int(scenario.get('frames') or frames)}",
        f"    custom uniform int gopex_inference_steps = {int(scenario.get('inference_steps') or 0)}",
        f"    custom uniform int gopex_render_seed = {int(scenario.get('seed') or 0)}",
        f"    custom uniform bool gopex_quality_ok = {'true' if evaluation.get('quality_ok') is True else 'false'}",
        f"    custom uniform string gopex_quality_failures = {_q(','.join(str(x) for x in evaluation.get('quality_failures') or []))}",
        f"    custom uniform string gopex_quality_report = {_q(str(evaluation.get('quality_report') or ''))}",
        f"    custom uniform string gopex_context_json = {_q(context_json)}",
        "",
        '    def Scope "Context" (',
        '        kind = "component"',
        "    )",
        "    {",
        f"        custom uniform string gopex_context_schema = {_q('gopex.usda_context/v1')}",
        f"        custom uniform string gopex_context_role = {_q('shot + scenario + training + evaluation')}",
        f"        custom uniform string gopex_prompt_hash = {_q(context['prompt']['sha256'])}",
        "    }",
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
