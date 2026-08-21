"""Qwen3-TTS path resolution, install plan, and dependency checks."""

from __future__ import annotations

import os
import shutil
from importlib import metadata
from pathlib import Path
from typing import Any

from ltx_trainer.qwen3_tts.config import GenerationMode, ModelChoice, Qwen3TtsConfig


def models_root() -> Path:
    env = os.environ.get("GOPEX_QWEN3_TTS_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        for sub in ("models/qwen-tts", "ComfyUI/models/qwen-tts"):
            candidate = parent / sub
            if candidate.is_dir():
                return candidate
    return Path("models/qwen-tts").resolve()


def voices_dir(root: Path | None = None) -> Path:
    return (root or models_root()) / "voices"


def _local_candidates(model_id: str, root: Path) -> list[Path]:
    """ComfyUI-Qwen-TTS layout: ``models/qwen-tts/Qwen/Qwen3-TTS-...``."""
    tail = model_id.split("/", 1)[-1]
    return [
        root / model_id,
        root / "Qwen" / tail,
        root / tail,
    ]


def resolve_model_path(model_id: str, root: Path | None = None) -> str:
    """Prefer local weights under ``models/qwen-tts``; else return Hub id."""
    root = root or models_root()
    for candidate in _local_candidates(model_id, root):
        if candidate.is_dir() and any(candidate.iterdir()):
            return str(candidate)
    return model_id


def check_transformers_version(cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    c = cfg or Qwen3TtsConfig()
    try:
        import transformers  # noqa: PLC0415

        version = transformers.__version__
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "ok": False,
            "required": c.transformers_pin,
            "message": f"transformers not installed; pin {c.transformers_pin} (not >=5.0)",
        }

    major = int(version.split(".", maxsplit=1)[0])
    ok = major < 5 and version.startswith(c.transformers_pin.rsplit(".", 1)[0])
    # Exact pin recommended by upstream; allow patch drift within 4.57.x
    if version.startswith("4.57."):
        ok = True
    if major >= 5:
        ok = False
    return {
        "installed": True,
        "version": version,
        "ok": ok,
        "required": c.transformers_pin,
        "message": (
            "transformers OK"
            if ok
            else f"transformers {version} incompatible — downgrade: pip install transformers=={c.transformers_pin}"
        ),
    }


def package_status(cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    c = cfg or Qwen3TtsConfig()
    root = models_root()
    try:
        qwen_ver = metadata.version(c.pip_package)
        qwen_installed = True
    except metadata.PackageNotFoundError:
        qwen_ver = None
        qwen_installed = False

    return {
        "pip_package": c.pip_package,
        "qwen_tts_installed": qwen_installed,
        "qwen_tts_version": qwen_ver,
        "transformers": check_transformers_version(c),
        "models_root": str(root),
        "models_root_exists": root.is_dir(),
        "voices_dir": str(voices_dir(root)),
        "hf_cache": os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE"),
    }


def install_plan(cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    c = cfg or Qwen3TtsConfig()
    root = models_root()
    return {
        "github": c.github,
        "pip_install": (
            f"pip install {c.pip_package} "
            f"transformers=={c.transformers_pin} accelerate=={c.accelerate_pin} "
            "torch torchaudio librosa soundfile"
        ),
        "optional_flash": "pip install flash-attn --no-build-isolation",
        "optional_sage": "pip install sageattention",
        "models_root": str(root),
        "download_example": [
            f"huggingface-cli download {c.hf_base_17b} --local-dir {root}/Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            f"huggingface-cli download {c.hf_voice_design} --local-dir {root}/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            f"huggingface-cli download {c.hf_custom_17b} --local-dir {root}/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            f"huggingface-cli download {c.hf_tokenizer} --local-dir {root}/Qwen/Qwen3-TTS-Tokenizer-12Hz",
        ],
        "gopex_env": {
            "GOPEX_QWEN3_TTS_ROOT": str(root),
        },
        "transformers_warning": "Qwen3-TTS breaks on transformers>=5.0 — pin 4.57.3",
    }


def build_infer_argv(
    *,
    mode: GenerationMode,
    text: str,
    output: str = "output.wav",
    instruct: str | None = None,
    speaker: str | None = None,
    language: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    model_choice: ModelChoice = "1.7B",
    attention: str = "auto",
    unload_model_after_generate: bool = False,
    top_p: float | None = None,
    top_k: int | None = None,
    temperature: float | None = None,
    repetition_penalty: float | None = None,
    script: str | None = None,
    role_bank_json: str | None = None,
    cfg: Qwen3TtsConfig | None = None,
) -> dict[str, Any]:
    """Return subprocess plan for ``qwen3_tts_infer.py``."""
    c = cfg or Qwen3TtsConfig()
    model_id = resolve_model_path(c.model_id_for(mode, model_choice))
    cmd = [
        "python",
        "-m",
        "ltx_trainer.qwen3_tts.infer_cli",
        "--mode",
        mode,
        "--text",
        text,
        "--output",
        output,
        "--model",
        model_id,
        "--model-choice",
        model_choice,
        "--attention",
        attention,
    ]
    if language:
        cmd.extend(["--language", language])
    if speaker:
        cmd.extend(["--speaker", speaker])
    if instruct:
        cmd.extend(["--instruct", instruct])
    if ref_audio:
        cmd.extend(["--ref-audio", ref_audio])
    if ref_text:
        cmd.extend(["--ref-text", ref_text])
    if script:
        cmd.extend(["--script", script])
    if role_bank_json:
        cmd.extend(["--role-bank", role_bank_json])
    if unload_model_after_generate:
        cmd.append("--unload")
    for flag, val in (
        ("--top-p", top_p),
        ("--top-k", top_k),
        ("--temperature", temperature),
        ("--repetition-penalty", repetition_penalty),
    ):
        if val is not None:
            cmd.extend([flag, str(val)])

    return {
        "mode": mode,
        "model_id": model_id,
        "command": cmd,
        "models_root": str(models_root()),
        "python_api_module": "ltx_trainer.qwen3_tts.runtime",
    }


def upstream_knowledge(cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    c = cfg or Qwen3TtsConfig()
    return {
        "title": c.title,
        "framework": c.framework,
        "license": c.license,
        "github": c.github,
        "comfy_reference": c.comfy_reference,
        "pip_package": c.pip_package,
        "transformers_pin": c.transformers_pin,
        "languages": list(c.languages),
        "speakers": list(c.speakers),
        "generation_modes": list(c.generation_modes),
        "attention_mechanisms": list(c.attention_mechanisms),
        "models": {
            "tokenizer": c.hf_tokenizer,
            "base_17b": c.hf_base_17b,
            "base_06b": c.hf_base_06b,
            "voice_design": c.hf_voice_design,
            "custom_17b": c.hf_custom_17b,
            "custom_06b": c.hf_custom_06b,
        },
        "install": install_plan(c),
        "status": package_status(c),
    }


def doctor(cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    c = cfg or Qwen3TtsConfig()
    status = package_status(c)
    issues: list[str] = []
    if not status["qwen_tts_installed"]:
        issues.append(f"pip package {c.pip_package} not installed")
    if not status["transformers"]["ok"]:
        issues.append(status["transformers"]["message"])
    if not shutil.which("python") and not shutil.which("python3"):
        issues.append("python interpreter not found on PATH")
    return {"ok": not issues, "issues": issues, "status": status, "install": install_plan(c)}
