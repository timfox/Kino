"""MMAE rubger backends: mock Table-2 rates, OpenAI text, or Qwen3-Omni multimodal."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ltx_trainer.mmae.audio_bundle import build_multimodal_user_content, clips_for_rubric
from ltx_trainer.mmae.config import RubricCategory
from ltx_trainer.mmae.judger import JUDGER_SYSTEM_PROMPT, format_user_prompt, mock_judger_choice
from ltx_trainer.mmae.sample import MMAESample, Rubric


@dataclass(frozen=True)
class JudgerConfig:
    mode: str = "mock"
    base_url: str | None = None
    model: str | None = None
    api_key: str = "EMPTY"
    timeout_sec: float = 120.0
    omni_part_style: str = "input_audio"
    omni_require_audio: bool = False
    note: str = ""


class RubricJudger(Protocol):
    def choose(self, rubric: Rubric, choices: list[str], *, seed: int) -> str: ...


@dataclass
class MockTableJudger:
    p_correct_if: float
    p_correct_cr: float

    def choose(self, rubric: Rubric, choices: list[str], *, seed: int) -> str:
        p = self.p_correct_if if rubric.category == RubricCategory.INSTRUCTION_FOLLOWING else self.p_correct_cr
        rng = random.Random(seed)
        return mock_judger_choice(rubric.right_choice, choices, p_correct=p, rng=rng)


@dataclass
class OpenAICompatTextJudger:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_sec: float = 120.0
    audio_note: str = ""

    def choose(self, rubric: Rubric, choices: list[str], *, seed: int) -> str:
        from openai import OpenAI

        question = rubric.question
        if self.audio_note:
            question = f"{question}\n\nAudio context (paths only; multimodal not wired):\n{self.audio_note}"
        user_prompt = format_user_prompt(question, choices)
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_sec)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            seed=seed % (2**31 - 1),
        )
        text = (resp.choices[0].message.content or "").strip()
        return parse_judger_choice_letter(text, len(choices))


@dataclass
class OmniSampleJudger:
    """Per-sample Qwen3-Omni judger with WAV attachments (Appendix C)."""

    base_url: str
    model: str
    sample: MMAESample
    dataset_root: Path | None
    predictions_dir: Path | None
    api_key: str = "EMPTY"
    timeout_sec: float = 120.0
    part_style: str = "input_audio"
    require_audio: bool = False
    slice_cache_dir: Path | None = None

    def choose(self, rubric: Rubric, choices: list[str], *, seed: int) -> str:
        from openai import OpenAI

        clips = clips_for_rubric(
            rubric,
            self.sample,
            dataset_root=self.dataset_root,
            predictions_dir=self.predictions_dir,
        )
        if self.require_audio and not clips:
            raise FileNotFoundError(
                f"no audio clips for sample {self.sample.sample_id!r} rubric: {rubric.question[:80]}"
            )
        content = build_multimodal_user_content(
            rubric,
            choices,
            clips,
            part_style=self.part_style,
            cache_dir=self.slice_cache_dir,
        )
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_sec)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGER_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
            seed=seed % (2**31 - 1),
        )
        text = (resp.choices[0].message.content or "").strip()
        return parse_judger_choice_letter(text, len(choices))


def parse_judger_choice_letter(text: str, num_choices: int) -> str:
    """Parse ``{"choice": "B"}`` or bare letter from judger output."""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "choice" in payload:
            letter = str(payload["choice"]).strip().upper()
            if len(letter) == 1 and ord("A") <= ord(letter) < ord("A") + num_choices:
                return letter
    except json.JSONDecodeError:
        pass
    match = re.search(r'"choice"\s*:\s*"([A-Z])"', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for ch in reversed(re.findall(r"\b([A-Z])\b", text.upper())):
        if ord("A") <= ord(ch) < ord("A") + num_choices:
            return ch
    return "A"


def _resolve_judger_alias() -> str | None:
    alias = os.environ.get("GOPEX_MMAE_JUDGER", "").strip().lower()
    if alias in ("qwen3-omni", "omni", "qwen-omni"):
        return "omni"
    if alias in ("mock", "openai", "omni"):
        return alias
    return None


def resolve_judger_config() -> JudgerConfig:
    alias_mode = _resolve_judger_alias()
    mode = os.environ.get("GOPEX_MMAE_JUDGER_MODE", "mock").strip().lower() or "mock"
    if alias_mode:
        mode = alias_mode

    part_style = os.environ.get("GOPEX_MMAE_OMNI_AUDIO_PART", "input_audio").strip() or "input_audio"
    require_audio = os.environ.get("GOPEX_MMAE_OMNI_REQUIRE_AUDIO", "").strip().lower() in ("1", "true", "yes")
    api_key = os.environ.get("GOPEX_MMAE_JUDGER_API_KEY", "EMPTY").strip() or "EMPTY"
    timeout = float(os.environ.get("GOPEX_MMAE_JUDGER_TIMEOUT_SEC", "120"))

    if mode == "omni":
        base = os.environ.get("GOPEX_MMAE_OMNI_BASE_URL", "").strip().rstrip("/")
        model = os.environ.get("GOPEX_MMAE_OMNI_MODEL", "").strip()
        if not base:
            base = os.environ.get("QWEN_BASE_URL", "").strip().rstrip("/") or None
        if not model:
            model = (
                os.environ.get("QWEN_MODEL", "").strip()
                or os.environ.get("GOPEX_CAPTION_VLM_MODEL", "").strip()
                or None
            )
        note = ""
        if not base:
            note = "GOPEX_MMAE_OMNI_BASE_URL or QWEN_BASE_URL unset; omni probe will fail until configured."
        return JudgerConfig(
            mode=mode,
            base_url=base,
            model=model,
            api_key=api_key,
            timeout_sec=timeout,
            omni_part_style=part_style,
            omni_require_audio=require_audio,
            note=note,
        )

    base = os.environ.get("GOPEX_MMAE_JUDGER_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("GOPEX_MMAE_JUDGER_MODEL", "").strip()
    if not base:
        base = os.environ.get("GEMMA4_BASE_URL", "").strip().rstrip("/") or None
    if not model:
        model = os.environ.get("GEMMA4_MODEL", "").strip() or None
    note = ""
    if mode == "openai" and not base:
        note = "GOPEX_MMAE_JUDGER_BASE_URL unset; probe will fail until vLLM is configured."
    return JudgerConfig(
        mode=mode,
        base_url=base,
        model=model,
        api_key=api_key,
        timeout_sec=timeout,
        omni_part_style=part_style,
        omni_require_audio=require_audio,
        note=note,
    )


def build_judger(
    *,
    mode: str | None = None,
    model_skill: tuple[float, float] | None = None,
) -> RubricJudger:
    cfg = resolve_judger_config()
    use_mode = (mode or cfg.mode).lower()
    if use_mode == "openai":
        if not cfg.base_url or not cfg.model:
            raise ValueError("openai judger requires GOPEX_MMAE_JUDGER_BASE_URL and GOPEX_MMAE_JUDGER_MODEL")
        return OpenAICompatTextJudger(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key,
            timeout_sec=cfg.timeout_sec,
        )
    if use_mode == "omni":
        raise ValueError("omni judger requires per-sample context; use OmniSampleJudger in evaluate_samples")
    p_if, p_cr = model_skill or (0.5, 0.5)
    return MockTableJudger(p_correct_if=p_if, p_correct_cr=p_cr)


def probe_judger(*, mode: str | None = None) -> dict[str, Any]:
    """Connectivity check for configured judger backend."""
    cfg = resolve_judger_config()
    use_mode = (mode or cfg.mode).lower()
    if use_mode == "mock":
        return {
            "ok": True,
            "mode": "mock",
            "note": "Mock judger uses Table 2 anchor rates; no network required.",
        }
    if not cfg.base_url or not cfg.model:
        return {
            "ok": False,
            "mode": use_mode,
            "error": "missing base_url or model",
            "config": {k: v for k, v in cfg.__dict__.items() if k != "api_key"},
        }
    try:
        from openai import OpenAI

        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=min(cfg.timeout_sec, 30.0))
        models = client.models.list()
        ids = [m.id for m in getattr(models, "data", [])][:5]
        note = (
            "Multimodal omni judger; rubrics attach WAV via input_audio parts."
            if use_mode == "omni"
            else "Text-only rubric probe; set GOPEX_MMAE_JUDGER_MODE=omni for paper-faithful audio eval."
        )
        return {
            "ok": True,
            "mode": use_mode,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "models_sample": ids,
            "omni_part_style": cfg.omni_part_style,
            "note": note,
        }
    except Exception as exc:  # noqa: BLE001 — probe reports error string
        return {
            "ok": False,
            "mode": use_mode,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "error": str(exc),
        }
