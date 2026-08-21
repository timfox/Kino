"""Inference argv builder for ID-LoRA submodule scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from ltx_trainer.id_lora.config import IdLoraConfig


def _root() -> Path:
    env = os.environ.get("GOPEX_ID_LORA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "ID-LoRA"
        if candidate.is_dir():
            return candidate
    return Path("ID-LoRA")


def _model_dir(root: Path) -> Path:
    custom = os.environ.get("GOPEX_ID_LORA_MODELS", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return root / "models"


def inference_defaults(cfg: IdLoraConfig | None = None) -> dict[str, Any]:
    c = cfg or IdLoraConfig()
    root = _root()
    models = _model_dir(root)
    return {
        "id_lora_root": str(root),
        "models_dir": str(models),
        "checkpoint_ltx2": str(models / "ltx-2-19b-dev.safetensors"),
        "text_encoder": str(models / "gemma-3-12b-it-qat-q4_0-unquantized"),
        "lora_celebvhq": str(models / "id-lora-celebvhq" / "lora_weights.safetensors"),
        "video_guidance_scale": c.default_video_guidance,
        "audio_guidance_scale": c.default_audio_guidance,
        "identity_guidance_scale": c.default_identity_guidance,
        "num_inference_steps": c.default_inference_steps,
        "height": c.default_resolution[0],
        "width": c.default_resolution[1],
        "num_frames": c.default_num_frames,
        "frame_rate": c.default_frame_rate,
    }


def build_inference_argv(
    mode: Literal["one_stage", "two_stage", "two_stage_hq"] = "one_stage",
    *,
    lora_path: str | None = None,
    reference_audio: str | None = None,
    first_frame: str | None = None,
    prompt: str | None = None,
    output_dir: str | None = None,
    ltx_version: Literal["2", "2.3"] = "2",
    cfg: IdLoraConfig | None = None,
) -> dict[str, Any]:
    """Return script path + argv for subprocess delegation."""
    c = cfg or IdLoraConfig()
    root = _root()
    defaults = inference_defaults(c)
    if ltx_version == "2.3":
        script_dir = root / "ID-LoRA-2.3" / "scripts"
        if mode == "two_stage_hq":
            script = script_dir / "inference_two_stage_hq.py"
        elif mode == "two_stage":
            script = script_dir / "inference_two_stage.py"
        else:
            script = script_dir / "inference_one_stage.py"
    else:
        script_dir = root / "scripts"
        script = script_dir / ("inference_two_stage.py" if mode == "two_stage" else "inference_one_stage.py")

    examples = root / "examples"
    argv = [
        str(script),
        "--lora-path",
        lora_path or defaults["lora_celebvhq"],
        "--reference-audio",
        reference_audio or str(examples / "reference.wav"),
        "--first-frame",
        first_frame or str(examples / "first_frame.png"),
        "--prompt",
        prompt or "",
        "--output-dir",
        output_dir or str(root / "outputs" / "gopex"),
        "--checkpoint",
        defaults["checkpoint_ltx2"],
        "--text-encoder-path",
        defaults["text_encoder"],
        "--video-guidance-scale",
        str(c.default_video_guidance),
        "--audio-guidance-scale",
        str(c.default_audio_guidance),
        "--identity-guidance-scale",
        str(c.default_identity_guidance),
        "--num-inference-steps",
        str(c.default_inference_steps),
    ]
    return {
        "mode": mode,
        "ltx_version": ltx_version,
        "script": str(script),
        "argv": argv,
        "cwd": str(root),
        "defaults": defaults,
    }
