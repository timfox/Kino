"""Persist voice clone prompts and speaker metadata (SaveVoice / LoadSpeaker parity)."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ltx_trainer.qwen3_tts.upstream import voices_dir


@dataclass
class SavedVoice:
    name: str
    ref_audio: str | None = None
    ref_text: str | None = None
    model_choice: str = "1.7B"
    language: str = "Auto"
    instruct: str | None = None
    speaker: str | None = None
    mode: str = "clone_prompt"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prompt_file: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _voice_json_path(root: Path, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return root / f"{safe}.voice.json"


def _prompt_pkl_path(root: Path, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return root / f"{safe}.prompt.pkl"


def save_voice(
    voice: SavedVoice,
    *,
    prompt_item: Any | None = None,
    root: Path | None = None,
) -> dict[str, str]:
    """Write ``.voice.json`` and optional ``.prompt.pkl`` under voices dir."""
    root = root or voices_dir()
    root.mkdir(parents=True, exist_ok=True)

    if prompt_item is not None:
        pkl = _prompt_pkl_path(root, voice.name)
        with pkl.open("wb") as fh:
            pickle.dump(prompt_item, fh)
        voice.prompt_file = str(pkl)

    json_path = _voice_json_path(root, voice.name)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(voice.to_dict(), fh, indent=2)
    return {"voice_json": str(json_path), "prompt_pkl": voice.prompt_file or ""}


def load_voice(name: str, *, root: Path | None = None) -> tuple[SavedVoice, Any | None]:
    root = root or voices_dir()
    json_path = _voice_json_path(root, name)
    if not json_path.is_file():
        raise FileNotFoundError(f"saved voice not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    voice = SavedVoice(**data)
    prompt_item = None
    prompt_path = data.get("prompt_file")
    if prompt_path and Path(prompt_path).is_file():
        with Path(prompt_path).open("rb") as fh:
            prompt_item = pickle.load(fh)  # noqa: S301 — local voice library only
    return voice, prompt_item


def list_saved_voices(*, root: Path | None = None) -> list[str]:
    root = root or voices_dir()
    if not root.is_dir():
        return []
    return sorted(p.stem.replace(".voice", "") for p in root.glob("*.voice.json"))
