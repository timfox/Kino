"""Upstream rednote-hilab/dots.tts integration — path resolution and argv builders."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Literal

from ltx_trainer.dotstts.config import DotsttsConfig

CheckpointKind = Literal["base", "soar", "mf"]
InferMode = Literal["cli", "gradio", "python"]


def upstream_root() -> Path:
    env = os.environ.get("GOPEX_DOTSTTS_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "dots.tts"
        if candidate.is_dir():
            return candidate
    return Path("dots.tts").resolve()


def upstream_venv_python(root: Path | None = None) -> Path | None:
    root = root or upstream_root()
    candidate = root / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def hf_checkpoints(cfg: DotsttsConfig | None = None) -> dict[str, str]:
    c = cfg or DotsttsConfig()
    return dict(c.hf_checkpoints)


def resolve_checkpoint(
    kind: CheckpointKind | None = None,
    *,
    model_name_or_path: str | None = None,
    cfg: DotsttsConfig | None = None,
) -> str:
    if model_name_or_path:
        return model_name_or_path
    c = cfg or DotsttsConfig()
    models = hf_checkpoints(c)
    key = kind or c.default_checkpoint
    if key not in models:
        raise KeyError(f"Unknown checkpoint kind {key!r}; expected one of {sorted(models)}")
    return models[key]


def upstream_status(root: Path | None = None) -> dict[str, Any]:
    root = root or upstream_root()
    pyproject = root / "pyproject.toml"
    package_dir = root / "src" / "dots_tts"
    cli_entry = shutil.which("dots.tts")
    venv_py = upstream_venv_python(root)
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "is_git_repo": (root / ".git").is_dir(),
        "has_pyproject": pyproject.is_file(),
        "has_package": package_dir.is_dir(),
        "venv_python": str(venv_py) if venv_py else None,
        "cli_on_path": cli_entry,
        "ready": pyproject.is_file() and package_dir.is_dir(),
    }


def install_plan(cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    c = cfg or DotsttsConfig()
    root = upstream_root()
    return {
        "github": c.github,
        "clone": f"git clone {c.github}.git {root}",
        "init_env": [
            f"cd {root}",
            "python -m venv .venv",
            "source .venv/bin/activate",
            "python -m pip install --upgrade pip",
            "python -m pip install -e . -c constraints/recommended.txt",
        ],
        "full_extras": "python -m pip install -e .[full] -c constraints/recommended.txt",
        "constraints": str(root / "constraints" / "recommended.txt"),
        "python_versions": ["3.10", "3.11", "3.12"],
        "hf_checkpoints": hf_checkpoints(c),
        "default_checkpoint": c.default_checkpoint,
        "gopex_env": {
            "GOPEX_DOTSTTS_ROOT": str(root),
            "GOPEX_DOTSTTS_MODEL": resolve_checkpoint(cfg=c),
        },
    }


def build_infer_argv(
    *,
    text: str,
    output: str = "output.wav",
    prompt_audio: str | None = None,
    prompt_text: str | None = None,
    checkpoint: CheckpointKind | None = None,
    model_name_or_path: str | None = None,
    num_steps: int | None = None,
    guidance_scale: float | None = None,
    language: str | None = None,
    normalize_text: bool = False,
    seed: int = 42,
    cfg: DotsttsConfig | None = None,
) -> dict[str, Any]:
    """Return subprocess plan for upstream ``dots.tts`` CLI."""
    c = cfg or DotsttsConfig()
    root = upstream_root()
    model = resolve_checkpoint(checkpoint, model_name_or_path=model_name_or_path, cfg=c)
    ckpt = checkpoint or c.default_checkpoint
    steps = num_steps if num_steps is not None else (c.mf_nfe if ckpt == "mf" else 10)
    guidance = guidance_scale if guidance_scale is not None else c.cfg_gamma

    argv = [
        "--model-name-or-path",
        model,
        "--text",
        text,
        "--output",
        output,
        "--seed",
        str(seed),
        "--num-steps",
        str(steps),
        "--guidance-scale",
        str(guidance),
    ]
    if prompt_audio:
        argv.extend(["--prompt-audio", prompt_audio])
    if prompt_text:
        argv.extend(["--prompt-text", prompt_text])
    if language:
        argv.extend(["--language", language])
    if normalize_text:
        argv.append("--normalize-text")

    venv_py = upstream_venv_python(root)
    cli_bin = root / ".venv" / "bin" / "dots.tts"
    if cli_bin.is_file():
        executable = str(cli_bin)
        full_argv = [str(cli_bin), *argv]
    elif shutil.which("dots.tts"):
        executable = "dots.tts"
        full_argv = ["dots.tts", *argv]
    else:
        python = str(venv_py) if venv_py else "python"
        executable = python
        full_argv = [python, "-m", "dots_tts.cli", *argv]

    return {
        "mode": "cli",
        "checkpoint": ckpt,
        "model_name_or_path": model,
        "executable": executable,
        "argv": full_argv,
        "cwd": str(root),
        "upstream_status": upstream_status(root),
    }


def build_gradio_argv(
    *,
    checkpoint: CheckpointKind | None = None,
    model_name_or_path: str | None = None,
    host: str = "0.0.0.0",
    port: int = 7860,
    optimize: bool = False,
    output_dir: str | None = None,
    cfg: DotsttsConfig | None = None,
) -> dict[str, Any]:
    c = cfg or DotsttsConfig()
    root = upstream_root()
    model = resolve_checkpoint(checkpoint, model_name_or_path=model_name_or_path, cfg=c)
    script = root / "apps" / "gradio" / "app.py"
    argv = [
        str(script),
        "--model-name-or-path",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if optimize:
        argv.append("--optimize")
    if output_dir:
        argv.extend(["--output-dir", output_dir])

    venv_py = upstream_venv_python(root)
    python = str(venv_py) if venv_py else "python"
    return {
        "mode": "gradio",
        "script": str(script),
        "executable": python,
        "argv": [python, *argv],
        "cwd": str(root),
        "url": f"http://{host}:{port}",
        "upstream_status": upstream_status(root),
    }


def build_prepare_data_argv(
    *,
    output_dir: str | None = None,
    cfg: DotsttsConfig | None = None,
) -> dict[str, Any]:
    root = upstream_root()
    out = output_dir or str(root / "downloaded_data")
    script = root / "scripts" / "prepare_train_jsonl_manifest.py"
    venv_py = upstream_venv_python(root)
    python = str(venv_py) if venv_py else "python"
    argv = [python, str(script), "--output-dir", out]
    return {
        "mode": "prepare_data",
        "script": str(script),
        "executable": python,
        "argv": argv,
        "cwd": str(root),
        "output_dir": out,
        "upstream_status": upstream_status(root),
    }


def build_train_argv(
    *,
    config: str | None = None,
    cfg: DotsttsConfig | None = None,
) -> dict[str, Any]:
    root = upstream_root()
    yaml_path = config or str(root / "configs" / "dots_tts.yaml")
    script = root / "scripts" / "train_dots_tts.py"
    venv_py = upstream_venv_python(root)
    python = str(venv_py) if venv_py else "python"
    argv = [
        "accelerate",
        "launch",
        str(script),
        "--config",
        yaml_path,
    ]
    return {
        "mode": "train",
        "script": str(script),
        "config": yaml_path,
        "executable": "accelerate",
        "argv": argv,
        "cwd": str(root),
        "note": "Smoke config — replace pretrained_model_path, data sources, and max_train_steps.",
        "upstream_status": upstream_status(root),
    }


def upstream_knowledge(cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    from ltx_trainer.dotstts.pipeline import framework_card, headline_results, training_stages

    c = cfg or DotsttsConfig()
    return {
        "framework": framework_card(c),
        "headline": headline_results(c),
        "training_stages": training_stages(),
        "hf_checkpoints": hf_checkpoints(c),
        "default_checkpoint": c.default_checkpoint,
        "upstream": upstream_status(),
        "install": install_plan(c),
        "quick_start": {
            "init": "./scripts/gopex-dotstts.sh init",
            "install": "./scripts/gopex-dotstts.sh install",
            "infer": "./scripts/gopex-dotstts.sh infer --text 'Hello' --prompt-audio ref.wav --prompt-text 'ref text'",
            "gradio": "./scripts/gopex-dotstts.sh gradio",
            "prepare_data": "./scripts/gopex-dotstts.sh prepare-data",
            "train_smoke": "./scripts/gopex-dotstts.sh train",
        },
    }
