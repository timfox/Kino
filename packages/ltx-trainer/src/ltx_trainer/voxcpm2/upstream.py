"""Upstream OpenBMB/VoxCPM integration — path resolution and argv builders."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Literal

from ltx_trainer.voxcpm2.config import Voxcpm2Config

InferMode = Literal["design", "clone", "batch", "python", "gradio"]


def upstream_root() -> Path:

    env = os.environ.get("GOPEX_VOXCPM2_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "VoxCPM"
        if candidate.is_dir():
            return candidate
    return Path("VoxCPM").resolve()


def resolve_model(model_name_or_path: str | None = None, cfg: Voxcpm2Config | None = None) -> str:
    if model_name_or_path:
        return model_name_or_path
    c = cfg or Voxcpm2Config()
    env = os.environ.get("GOPEX_VOXCPM2_MODEL", "").strip()
    return env or c.hf_model


def upstream_status(root: Path | None = None) -> dict[str, Any]:
    root = root or upstream_root()
    pyproject = root / "pyproject.toml"
    setup_py = root / "setup.py"
    app_py = root / "app.py"
    cli_entry = shutil.which("voxcpm")
    venv_py = root / ".venv" / "bin" / "python"
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "is_git_repo": (root / ".git").is_dir(),
        "has_pyproject": pyproject.is_file(),
        "has_setup_py": setup_py.is_file(),
        "has_app_py": app_py.is_file(),
        "venv_python": str(venv_py) if venv_py.is_file() else None,
        "cli_on_path": cli_entry,
        "pip_package": (Voxcpm2Config()).pip_package,
        "ready": pyproject.is_file() or setup_py.is_file() or cli_entry is not None,
    }


def install_plan(cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    root = upstream_root()
    return {
        "github": c.github,
        "hf_model": c.hf_model,
        "pip_install": f"pip install {c.pip_package}",
        "clone": f"git clone {c.upstream_clone_url} {root}",
        "python_api": [
            "from voxcpm import VoxCPM",
            "model = VoxCPM.from_pretrained('openbmb/VoxCPM2', load_denoiser=False)",
            "wav = model.generate(text='Hello', cfg_value=2.0, inference_timesteps=10)",
        ],
        "requirements": "Python >=3.10,<3.13; PyTorch >=2.5; CUDA >=12.0",
        "nano_vllm": "pip install nano-vllm-voxcpm",
        "vllm_omni": "vllm serve openbmb/VoxCPM2 --omni --port 8000",
        "gopex_env": {
            "GOPEX_VOXCPM2_ROOT": str(root),
            "GOPEX_VOXCPM2_MODEL": c.hf_model,
        },
    }


def build_infer_argv(
    *,
    text: str,
    output: str = "output.wav",
    mode: InferMode = "design",
    reference_audio: str | None = None,
    prompt_audio: str | None = None,
    prompt_text: str | None = None,
    control: str | None = None,
    cfg_value: float | None = None,
    inference_timesteps: int | None = None,
    model_name_or_path: str | None = None,
    cfg: Voxcpm2Config | None = None,
) -> dict[str, Any]:
    """Return subprocess plan for upstream ``voxcpm`` CLI or Python API."""
    c = cfg or Voxcpm2Config()
    model = resolve_model(model_name_or_path, cfg=c)
    cfg_val = cfg_value if cfg_value is not None else c.cfg_alpha
    steps = inference_timesteps if inference_timesteps is not None else c.inference_timesteps

    if mode == "clone":
        cmd = [
            "voxcpm",
            "clone",
            "--text",
            text,
            "--output",
            output,
        ]
        if reference_audio:
            cmd.extend(["--reference-audio", reference_audio])
        if prompt_audio:
            cmd.extend(["--prompt-audio", prompt_audio])
        if prompt_text:
            cmd.extend(["--prompt-text", prompt_text])
    elif mode == "batch":
        cmd = ["voxcpm", "batch", "--input", text, "--output-dir", output]
    else:
        cmd = ["voxcpm", "design", "--text", text, "--output", output]
        if control:
            cmd.extend(["--control", control])

    return {
        "mode": mode,
        "command": cmd,
        "python_api": {
            "model": model,
            "text": text,
            "reference_wav_path": reference_audio,
            "prompt_wav_path": prompt_audio,
            "prompt_text": prompt_text,
            "cfg_value": cfg_val,
            "inference_timesteps": steps,
        },
        "upstream_root": str(upstream_root()),
    }


def build_gradio_argv(*, port: int = 8808, device: str = "auto") -> dict[str, Any]:
    root = upstream_root()
    return {
        "command": ["python", "app.py", "--port", str(port), "--device", device],
        "cwd": str(root),
        "url": f"http://localhost:{port}",
    }


def upstream_knowledge(cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "license": c.license,
        "github": c.github,
        "hf_model": c.hf_model,
        "hf_demo": c.hf_demo,
        "docs": c.docs,
        "params_b": c.params_b,
        "languages": c.n_languages,
        "dialects": c.n_chinese_dialects,
        "training_hours_m": c.training_hours_m,
        "output_hz": c.decode_sample_rate_hz,
        "lm_token_rate_hz": c.lm_token_rate_hz,
        "capabilities": list(c.generation_modes),
        "install": install_plan(c),
        "status": upstream_status(),
    }
