#!/usr/bin/env python3
"""Unified CLI for speech/audio paper stubs (mustbench, comet, ag_repa, …)."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

_STUBS: tuple[str, ...] = (
    "mustbench",
    "comet",
    "btscafe",
    "voicegiraffe",
    "cafnet",
    "dlmasr",
    "cryacc",
    "childvox",
    "demon",
    "dasheng_audiogen",
    "ag_repa",
    "wavenext2",
    "robustspeechflow",
    "speech_quality_emb",
)

_CONFIG_CLASS: dict[str, str] = {
    "mustbench": "MustBenchConfig",
    "comet": "CometConfig",
    "btscafe": "BTSCafeConfig",
    "voicegiraffe": "VoiceGiraffeConfig",
    "cafnet": "CafNetConfig",
    "dlmasr": "DlmAsrConfig",
    "cryacc": "CryAccConfig",
    "childvox": "ChildVoxConfig",
    "demon": "DemonConfig",
    "dasheng_audiogen": "DashengAudioGenConfig",
    "ag_repa": "AgRepaConfig",
    "wavenext2": "Wavenext2Config",
    "robustspeechflow": "RobustSpeechFlowConfig",
    "speech_quality_emb": "SpeechQualityEmbConfig",
}

_KNOWLEDGE_FN: dict[str, str] = {
    "ag_repa": "knowledge_card",
}


def _load_cfg(stub: str) -> Any:
    mod = importlib.import_module(f"ltx_trainer.{stub}.config")
    return getattr(mod, _CONFIG_CLASS[stub])()


def _load_pipe(stub: str) -> Any:
    return importlib.import_module(f"ltx_trainer.{stub}.pipeline")


def _load_mock(stub: str) -> Any:
    return importlib.import_module(f"ltx_trainer.{stub}.mock")


def _call_bundle(fn: Callable[..., Any], cfg: Any) -> Any:
    try:
        return fn(cfg)
    except TypeError:
        return fn()


def _cmd_knowledge(stub: str) -> int:
    pipe = _load_pipe(stub)
    cfg = _load_cfg(stub)
    fn_name = _KNOWLEDGE_FN.get(stub, "framework_card")
    print(json.dumps(getattr(pipe, fn_name)(cfg), indent=2, default=str))
    return 0


def _cmd_framework(stub: str) -> int:
    pipe = _load_pipe(stub)
    cfg = _load_cfg(stub)
    print(json.dumps(pipe.framework_card(cfg), indent=2, default=str))
    return 0


def _cmd_tables(stub: str) -> int:
    pipe = _load_pipe(stub)
    cfg = _load_cfg(stub)
    print(json.dumps(_call_bundle(pipe.benchmarks_bundle, cfg), indent=2, default=str))
    return 0


def _cmd_smoke(stub: str) -> int:
    mock = _load_mock(stub)
    cfg = _load_cfg(stub)
    try:
        out = mock.evaluation_smoke(cfg)
    except TypeError:
        out = mock.evaluation_smoke()
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_demo(stub: str, *, seed: int) -> int:
    pipe = _load_pipe(stub)
    cfg = _load_cfg(stub)
    if stub == "ag_repa":
        out = pipe.evaluation_demo(strategy="ag_repa", cfg=cfg)
    elif hasattr(pipe, "pipeline_demo"):
        try:
            out = pipe.pipeline_demo(cfg, seed=seed)
        except TypeError:
            out = pipe.pipeline_demo(seed=seed)
    else:
        out = pipe.evaluation_demo(cfg)
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_eval(stub: str, *, seed: int) -> int:
    pipe = _load_pipe(stub)
    cfg = _load_cfg(stub)
    if not hasattr(pipe, "evaluation_demo"):
        print(json.dumps({"error": f"{stub} has no evaluation_demo"}, indent=2))
        return 1
    try:
        out = pipe.evaluation_demo(seed=seed)
    except TypeError:
        try:
            out = pipe.evaluation_demo(cfg)
        except TypeError:
            out = pipe.evaluation_demo()
    print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Speech/audio paper stub CLI")
    p.add_argument("stub", choices=_STUBS)
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("knowledge", "framework", "tables", "smoke"):
        sp = sub.add_parser(name)
        sp.set_defaults(handler=name)

    sp = sub.add_parser("demo")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(handler="demo")

    sp = sub.add_parser("eval")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(handler="eval")

    args = p.parse_args(argv)
    stub = args.stub
    if args.handler == "knowledge":
        return _cmd_knowledge(stub)
    if args.handler == "framework":
        return _cmd_framework(stub)
    if args.handler == "tables":
        return _cmd_tables(stub)
    if args.handler == "smoke":
        return _cmd_smoke(stub)
    if args.handler == "demo":
        return _cmd_demo(stub, seed=args.seed)
    return _cmd_eval(stub, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
