"""Optional vLLM backend for Role-Agent WIA state prediction and AIW analysis."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ltx_trainer.role_agent.aiw import FailureReflection, parse_reflection, parse_selected_tasks
from ltx_trainer.role_agent.prompts import (
    FAILURE_ABSTRACTION_TEMPLATE,
    STATE_PREDICTION_TEMPLATE,
    TASK_RETRIEVAL_TEMPLATE,
)
from ltx_trainer.role_agent.state_predictor import PredictFn, parse_predicted_state


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def role_agent_llm_enabled() -> bool:
    return _env_flag("GOPEX_ROLE_AGENT_LLM")


def role_agent_wia_llm_enabled() -> bool:
    return _env_flag("GOPEX_ROLE_AGENT_WIA_LLM") or role_agent_llm_enabled()


def role_agent_aiw_llm_enabled() -> bool:
    return _env_flag("GOPEX_ROLE_AGENT_AIW_LLM") or role_agent_llm_enabled()


def role_agent_llm_rollout_enabled() -> bool:
    return _env_flag("GOPEX_ROLE_AGENT_LLM_ROLLOUT") or role_agent_llm_enabled()


@dataclass(frozen=True)
class RoleAgentLLMConfig:
    base_url: str
    model: str
    api_key: str
    timeout_s: float
    rollout_temperature: float
    reflection_temperature: float
    max_tokens: int

    @classmethod
    def from_env(cls) -> RoleAgentLLMConfig:
        base = (
            os.environ.get("GOPEX_ROLE_AGENT_VLLM_URL")
            or os.environ.get("GOPEX_VLLM_QWEN_URL")
            or "http://127.0.0.1:8002/v1"
        ).strip().rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1" if "/v1" not in base else base
        model = (
            os.environ.get("GOPEX_ROLE_AGENT_MODEL")
            or os.environ.get("GOPEX_VLLM_QWEN_MODEL")
            or "Qwen/Qwen2.5-7B-Instruct"
        ).strip()
        return cls(
            base_url=base,
            model=model,
            api_key=os.environ.get("GOPEX_ROLE_AGENT_API_KEY", "unused").strip() or "unused",
            timeout_s=float(os.environ.get("GOPEX_ROLE_AGENT_LLM_TIMEOUT", "60")),
            rollout_temperature=float(os.environ.get("GOPEX_ROLE_AGENT_ROLLOUT_TEMP", "0.9")),
            reflection_temperature=float(os.environ.get("GOPEX_ROLE_AGENT_REFLECTION_TEMP", "0.5")),
            max_tokens=int(os.environ.get("GOPEX_ROLE_AGENT_LLM_MAX_TOKENS", "512")),
        )


@dataclass
class RoleAgentLLMClient:
    config: RoleAgentLLMConfig

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        cfg = self.config
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.rollout_temperature if temperature is None else temperature,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
        }
        url = f"{cfg.base_url.rstrip('/')}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s, context=ctx) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {url}: {body[:400]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"vLLM unreachable at {url}: {e}") from e
        obj = json.loads(raw)
        try:
            content = obj["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected chat response: {raw[:400]}") from e
        return str(content or "").strip()

    def complete(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )


def default_llm_client() -> RoleAgentLLMClient | None:
    if not (role_agent_llm_enabled() or role_agent_wia_llm_enabled() or role_agent_aiw_llm_enabled()):
        return None
    return RoleAgentLLMClient(RoleAgentLLMConfig.from_env())


def probe_vllm(client: RoleAgentLLMClient | None = None) -> dict[str, Any]:
    """Health probe for local vLLM (models list + optional one-token ping)."""
    cfg = (client or RoleAgentLLMClient(RoleAgentLLMConfig.from_env())).config
    base = cfg.base_url.rstrip("/")
    models_url = f"{base}/models"
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    out: dict[str, Any] = {
        "base_url": cfg.base_url,
        "model": cfg.model,
        "reachable": False,
        "models": [],
        "chat_ok": False,
    }
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(models_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=min(cfg.timeout_s, 15.0), context=ctx) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        out["reachable"] = True
        out["models"] = [m.get("id") for m in obj.get("data", []) if m.get("id")]
    except Exception as e:
        out["error"] = str(e)
        return out
    try:
        c = RoleAgentLLMClient(cfg)
        reply = c.complete("Reply with OK only.", temperature=0.0, max_tokens=8)
        out["chat_ok"] = bool(reply)
        out["chat_sample"] = reply[:80]
    except Exception as e:
        out["chat_error"] = str(e)
    return out


def make_vllm_state_predictor(client: RoleAgentLLMClient) -> PredictFn:
    """Build PredictFn that calls vLLM with STATE_PREDICTION_TEMPLATE."""

    def _predict(state: str, action: str, horizon: int) -> str:
        prompt = STATE_PREDICTION_TEMPLATE.format(state=state, action=action, horizon=horizon)
        text = client.complete(prompt, temperature=client.config.rollout_temperature, max_tokens=256)
        return parse_predicted_state(text)

    return _predict


def llm_analyze_failure(
    client: RoleAgentLLMClient,
    *,
    task: str,
    trajectory: str,
    task_id: str,
    domain: str,
) -> FailureReflection | None:
    prompt = FAILURE_ABSTRACTION_TEMPLATE.format(task=task, trajectory=trajectory)
    text = client.complete(prompt, temperature=client.config.reflection_temperature, max_tokens=512)
    return parse_reflection(text, task_id=task_id, domain=domain)


def llm_retrieve_task_indices(
    client: RoleAgentLLMClient,
    reflection: FailureReflection,
    candidates: list[str],
) -> list[int]:
    prompt = TASK_RETRIEVAL_TEMPLATE.format(
        error_pattern=f"{reflection.dominant_type}: {reflection.core_lesson}",
        candidates_text="\n".join(f"[{i}] {c}" for i, c in enumerate(candidates)),
        mode_library=reflection.dominant_type,
    )
    text = client.complete(prompt, temperature=client.config.reflection_temperature, max_tokens=384)
    return parse_selected_tasks(text)


def try_make_state_predictor() -> PredictFn | None:
    if not role_agent_wia_llm_enabled():
        return None
    try:
        client = RoleAgentLLMClient(RoleAgentLLMConfig.from_env())
        return make_vllm_state_predictor(client)
    except Exception:
        return None
