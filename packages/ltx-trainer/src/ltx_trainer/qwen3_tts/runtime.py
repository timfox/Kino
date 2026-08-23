"""Optional runtime wrapper around official ``qwen-tts`` package."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.qwen3_tts.attention import resolve_attention
from ltx_trainer.qwen3_tts.config import GenerationMode, ModelChoice, Qwen3TtsConfig
from ltx_trainer.qwen3_tts.dialogue import DialogueLine, RoleBank, parse_dialogue_script
from ltx_trainer.qwen3_tts.upstream import resolve_model_path


@dataclass
class GenerationParams:
    top_p: float = 0.8
    top_k: int = 20
    temperature: float = 1.0
    repetition_penalty: float = 1.0

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
        }


class Qwen3TtsRuntime:
    """Lazy model cache with attention-specific keys (ComfyUI parity)."""

    _cache: dict[tuple[str, str], Any] = {}
    _deps: tuple[Any, Any] | None = None

    def __init__(self, cfg: Qwen3TtsConfig | None = None) -> None:
        self.cfg = cfg or Qwen3TtsConfig()
        self._last_attn: str | None = None

    @staticmethod
    def _require_qwen_tts():
        if Qwen3TtsRuntime._deps is not None:
            return Qwen3TtsRuntime._deps
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"PyTorch import failed: {exc}"
            ) from exc
        try:
            from qwen_tts import Qwen3TTSModel  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"qwen-tts import failed: {exc}"
            ) from exc
        Qwen3TtsRuntime._deps = (torch, Qwen3TTSModel)
        return Qwen3TtsRuntime._deps

    def _load(self, model_id: str, attention: str = "auto") -> Any:
        resolved_attn, attn_impl, warning = resolve_attention(attention)  # type: ignore[arg-type]
        if warning:
            print(f"⚠️ [Qwen3-TTS] {warning}")

        cache_key = (model_id, attn_impl)
        if cache_key in self._cache:
            return self._cache[cache_key]

        torch, Qwen3TTSModel = self._require_qwen_tts()

        path = resolve_model_path(model_id)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        model = Qwen3TTSModel.from_pretrained(
            path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        self._cache[cache_key] = model
        self._last_attn = resolved_attn
        print(f"✅ [Qwen3-TTS] Loaded {path} attn={resolved_attn} ({attn_impl})")
        return model

    def unload_all(self) -> None:
        if not self._cache:
            return
        print(f"🗑️ [Qwen3-TTS] Unloading {len(self._cache)} cached model(s)...")
        self._cache.clear()
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print("✅ [Qwen3-TTS] Model cache and GPU memory cleared")

    def _gen_params(self, params: GenerationParams | None) -> dict[str, Any]:
        p = params or GenerationParams(
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            temperature=self.cfg.temperature,
            repetition_penalty=self.cfg.repetition_penalty,
        )
        return p.as_kwargs()

    def generate_custom_voice(
        self,
        *,
        text: str,
        speaker: str,
        language: str = "Auto",
        instruct: str | None = None,
        model_choice: ModelChoice = "1.7B",
        attention: str = "auto",
        params: GenerationParams | None = None,
        unload: bool = False,
    ) -> tuple[list[np.ndarray], int]:
        model_id = self.cfg.model_id_for("custom_voice", model_choice)
        model = self._load(model_id, attention)
        lang = None if language in ("Auto", "") else language
        kwargs = self._gen_params(params)
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=lang or "Auto",
            speaker=speaker,
            instruct=instruct or "",
            **kwargs,
        )
        if unload:
            self.unload_all()
        return wavs, sr

    def generate_voice_design(
        self,
        *,
        text: str,
        instruct: str,
        language: str = "Auto",
        attention: str = "auto",
        params: GenerationParams | None = None,
        unload: bool = False,
    ) -> tuple[list[np.ndarray], int]:
        model_id = self.cfg.model_id_for("voice_design")
        model = self._load(model_id, attention)
        lang = None if language in ("Auto", "") else language
        kwargs = self._gen_params(params)
        wavs, sr = model.generate_voice_design(
            text=text,
            language=lang or "Auto",
            instruct=instruct,
            **kwargs,
        )
        if unload:
            self.unload_all()
        return wavs, sr

    def create_voice_clone_prompt(
        self,
        *,
        ref_audio: str,
        ref_text: str | None = None,
        model_choice: ModelChoice = "1.7B",
        attention: str = "auto",
        x_vector_only_mode: bool = False,
        unload: bool = False,
    ) -> Any:
        model_id = self.cfg.model_id_for("voice_clone", model_choice)
        model = self._load(model_id, attention)
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        if unload:
            self.unload_all()
        return prompt

    def generate_voice_clone(
        self,
        *,
        text: str | list[str],
        language: str | list[str] = "Auto",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        voice_clone_prompt: Any | list[Any] | None = None,
        model_choice: ModelChoice = "1.7B",
        attention: str = "auto",
        params: GenerationParams | None = None,
        unload: bool = False,
    ) -> tuple[list[np.ndarray], int]:
        model_id = self.cfg.model_id_for("voice_clone", model_choice)
        model = self._load(model_id, attention)
        kwargs = self._gen_params(params)

        def _norm_lang(lang: str) -> str:
            return "Auto" if lang in ("Auto", "") else lang

        if isinstance(text, list):
            if isinstance(language, list):
                langs = [_norm_lang(x) for x in language]
            else:
                langs = [_norm_lang(language)] * len(text)
        else:
            langs = _norm_lang(language)  # type: ignore[assignment]

        if voice_clone_prompt is not None:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=langs,
                voice_clone_prompt=voice_clone_prompt,
                **kwargs,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=langs,
                ref_audio=ref_audio,
                ref_text=ref_text,
                **kwargs,
            )
        if unload:
            self.unload_all()
        return wavs, sr

    def generate_dialogue(
        self,
        *,
        script: str,
        role_bank: RoleBank,
        model_choice: ModelChoice = "1.7B",
        attention: str = "auto",
        pause_seconds: float | None = None,
        merge_outputs: bool = True,
        batch_size: int | None = None,
        language: str = "Auto",
        params: GenerationParams | None = None,
        unload: bool = False,
    ) -> tuple[list[np.ndarray], int, list[DialogueLine]]:
        lines = parse_dialogue_script(script)
        pause = self.cfg.dialogue_pause_seconds if pause_seconds is None else pause_seconds
        batch = self.cfg.dialogue_batch_size if batch_size is None else batch_size
        lang = language

        segments: list[np.ndarray] = []
        sr_out: int | None = None

        for start in range(0, len(lines), max(1, batch)):
            chunk = lines[start : start + batch]
            texts = [ln.text for ln in chunk]
            langs = [lang] * len(chunk)
            prompts = [role_bank.get(ln.role) for ln in chunk]
            wavs, sr = self.generate_voice_clone(
                text=texts,
                language=langs,
                voice_clone_prompt=prompts,
                model_choice=model_choice,
                attention=attention,
                params=params,
                unload=False,
            )
            sr_out = sr
            for wav in wavs:
                segments.append(np.asarray(wav, dtype=np.float32))

        assert sr_out is not None
        if merge_outputs and segments:
            silence = np.zeros(int(sr_out * pause), dtype=np.float32)
            merged: list[np.ndarray] = []
            for i, seg in enumerate(segments):
                merged.append(seg)
                if i + 1 < len(segments):
                    merged.append(silence)
            return [np.concatenate(merged)], sr_out, lines

        if unload:
            self.unload_all()
        return segments, sr_out, lines


def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf  # noqa: PLC0415

    sf.write(path, wav, sr)
