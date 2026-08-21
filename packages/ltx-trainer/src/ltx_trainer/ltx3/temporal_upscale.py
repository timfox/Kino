"""Temporal upscaler integration (LTX 3.0 track D)."""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def _comfyui_temporal_fallback() -> Path:
    return Path.home() / "ComfyUI/models/latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors"


def default_temporal_upsampler_path() -> Path:
    env = os.environ.get("GOPEX_TEMPORAL_UPSAMPLER", "").strip()
    if env:
        return Path(env).expanduser()
    try:
        from gopex_ltx.native_assets import resolve_native_assets

        assets = resolve_native_assets()
        if assets.temporal_upsampler:
            return assets.temporal_upsampler
    except ImportError:
        pass
    return _comfyui_temporal_fallback()


def latent_frames_after_temporal_x2(latent_frames: int) -> int:
    """LatentUpsampler temporal path: T → 2T, then drop first frame."""
    return max(1, 2 * int(latent_frames) - 1)


def temporal_upscale_status() -> dict:
    """Report weight availability and wiring state for the temporal upscaler."""
    weight = default_temporal_upsampler_path()
    repo = Path(os.environ.get("GOPEX_REPO", ".")).resolve()
    pipeline = repo / "pipeline/temporal_upscale_kino.py"
    return {
        "weight_path": str(weight),
        "weight_present": weight.is_file(),
        "env_override": "GOPEX_TEMPORAL_UPSAMPLER",
        "pipeline_module": str(pipeline),
        "pipeline_present": pipeline.is_file(),
        "pipeline_status": "wired" if pipeline.is_file() else "missing",
        "doc": "documents/LTX_3_VISION.md#phase-ladder",
        "integration_steps": [
            "Encode segment or concat MP4 → VAE latent (pipeline/temporal_upscale_kino.py).",
            "Apply ltx-2.3-temporal-upscaler-x2 in latent space (VideoTemporalUpsampler).",
            "Decode with VideoDecoder; mux source audio when --with-audio (default).",
            "Optional: distilled stage-2 DiT refine on upsampled latents (future HQ pass).",
        ],
        "related_weight": "ltx-2.3-temporal-upscaler-x2-1.0.safetensors (Lightricks HF)",
        "minute_render_hook": "./scripts/kino-ltx3-minute-render.sh temporal-run",
    }


def temporal_upscale_cli_argv(
    *,
    input_mp4: str,
    output_mp4: str,
    checkpoint_path: str | None = None,
    temporal_upsampler_path: str | None = None,
    slow_motion: bool = False,
    no_audio: bool = False,
) -> list[str]:
    """Argv vector for pipeline/temporal_upscale_kino.py (testable without GPU)."""
    repo = Path(os.environ.get("GOPEX_REPO", ".")).resolve()
    ckpt = checkpoint_path or os.environ.get("NATIVE_LTX", "")
    ups = temporal_upsampler_path or str(default_temporal_upsampler_path())
    argv = [
        "python",
        str(repo / "pipeline/temporal_upscale_kino.py"),
        "--input",
        input_mp4,
        "--output",
        output_mp4,
        "--checkpoint-path",
        ckpt,
        "--temporal-upsampler-path",
        ups,
    ]
    if slow_motion:
        argv.append("--slow-motion")
    if no_audio:
        argv.append("--no-audio")
    return argv


def temporal_upscale_shell_command(**kwargs) -> str:
    return " ".join(shlex.quote(p) for p in temporal_upscale_cli_argv(**kwargs))


def temporal_upscale_post_concat_command(*, input_mp4: str, output_mp4: str) -> dict:
    """CLI step after kino-ltx3-minute-render.sh concat."""
    st = temporal_upscale_status()
    cmd = temporal_upscale_shell_command(input_mp4=input_mp4, output_mp4=output_mp4)
    return {
        "status": "ready" if st["pipeline_present"] else "missing_pipeline",
        "input": input_mp4,
        "output": output_mp4,
        "weight_present": st["weight_present"],
        "pipeline_status": st["pipeline_status"],
        "command": cmd,
        "note": "Run after concat when GOPEX_LTX3_TEMPORAL=1 or manual temporal-run.",
    }
