"""Paper editor registry and local baseline submission builders."""

from __future__ import annotations

import os
import random
import shutil
import struct
import wave
from pathlib import Path
from typing import Any, Callable

from ltx_trainer.mmae.benchmarks import TABLE2_OVERALL
from ltx_trainer.mmae.sample import MMAESample

BaselineBuilder = Callable[..., dict[str, Any]]

EDITOR_ENV_COMMANDS: dict[str, str] = {
    "Step-Audio-EditX": "${GOPEX_MMAE_STEP_AUDIO_EDITX_CMD:-step-audio-editx batch}",
    "Audio-Omni": "${GOPEX_MMAE_AUDIO_OMNI_CMD:-audio-omni batch}",
    "Ming-UniAudio": "${GOPEX_MMAE_MING_UNIAUDIO_CMD:-ming-uniaudio batch}",
    "MMEdit": "${GOPEX_MMAE_MMEDIT_CMD:-mmedit batch}",
    "SmartDJ w/o planner": "${GOPEX_MMAE_SMARTDJ_CMD:-smartdj batch --no-planner}",
    "SmartDJ w/ planner": "${GOPEX_MMAE_SMARTDJ_PLANNER_CMD:-smartdj batch --planner}",
}

EDITOR_REGISTRY: dict[str, dict[str, Any]] = {
    "Step-Audio-EditX": {
        "kind": "instruction_editor",
        "paper_ifr": 44.86,
        "submission": "external",
        "note": "Run upstream Step-Audio-EditX; write {sample_id}/output.wav per MMAE layout.",
    },
    "Audio-Omni": {
        "kind": "instruction_editor",
        "paper_ifr": 50.73,
        "submission": "external",
        "note": "Run upstream Audio-Omni editor; evaluate with omni judger.",
    },
    "Ming-UniAudio": {"kind": "instruction_editor", "paper_ifr": 29.82, "submission": "external"},
    "MMEdit": {"kind": "instruction_editor", "paper_ifr": 43.12, "submission": "external"},
    "SmartDJ w/o planner": {"kind": "planner_editor", "paper_ifr": 38.20, "submission": "external"},
    "SmartDJ w/ planner": {"kind": "planner_editor", "paper_ifr": 42.26, "submission": "external"},
    "Identity": {
        "kind": "baseline",
        "paper_ifr": 27.37,
        "submission": "local",
        "builder": "identity",
        "note": "Pass-through: output.wav equals input1.wav.",
    },
    "Noise": {
        "kind": "baseline",
        "paper_ifr": 32.08,
        "submission": "local",
        "builder": "noise",
        "note": "Heavy AWGN on input1 (paper Noise baseline proxy).",
    },
}

LOCAL_BASELINES = ("Identity", "Noise")


def editors_card() -> dict[str, Any]:
    return {
        "models": list(EDITOR_REGISTRY.keys()),
        "local_baselines": list(LOCAL_BASELINES),
        "table2_anchors": dict(TABLE2_OVERALL),
        "registry": EDITOR_REGISTRY,
    }


def table2_delta(model: str, rates_pct: dict[str, float]) -> dict[str, float] | None:
    anchor = TABLE2_OVERALL.get(model)
    if anchor is None:
        return None
    return {key: round(float(rates_pct[key]) - float(anchor[key]), 2) for key in ("IFR", "CR", "EMR")}


def _sample_seed(sample_id: str, offset: int = 0) -> int:
    value = offset
    for ch in sample_id:
        value = (value * 131 + ord(ch)) & 0xFFFFFFFF
    return value


def _write_noisy_wav(
    src: Path,
    dst: Path,
    *,
    snr_db: float = 8.0,
    seed: int = 0,
) -> None:
    with wave.open(str(src), "rb") as reader:
        params = reader.getparams()
        if params.comptype != "NONE" or params.sampwidth != 2:
            shutil.copy2(src, dst)
            return
        raw = reader.readframes(params.nframes)
    count = len(raw) // 2
    if count == 0:
        shutil.copy2(src, dst)
        return
    samples = struct.unpack(f"<{count}h", raw)
    sig_power = sum(s * s for s in samples) / count
    noise_std = (sig_power / max(10 ** (snr_db / 10.0), 1e-9)) ** 0.5
    rng = random.Random(seed)
    noisy = [
        max(-32768, min(32767, int(sample + rng.gauss(0.0, noise_std))))
        for sample in samples
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as writer:
        writer.setnchannels(params.nchannels)
        writer.setsampwidth(params.sampwidth)
        writer.setframerate(params.framerate)
        writer.setcomptype(params.comptype, params.compname)
        writer.writeframes(struct.pack(f"<{len(noisy)}h", *noisy))


def materialize_identity_baseline(
    predictions_dir: str | Path,
    samples: list[MMAESample],
    *,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Copy each sample's first input WAV to ``{predictions_dir}/{id}/output.wav``."""
    pred_root = Path(predictions_dir)
    pred_root.mkdir(parents=True, exist_ok=True)
    ds_root = Path(dataset_root)
    copied: list[str] = []
    missing: list[str] = []
    for sample in samples:
        if not sample.audio_paths:
            missing.append(sample.sample_id)
            continue
        src = ds_root / sample.audio_paths[0]
        if not src.is_file():
            missing.append(sample.sample_id)
            continue
        out_dir = pred_root / sample.sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / "output.wav"
        shutil.copy2(src, dst)
        copied.append(sample.sample_id)
    return {
        "baseline": "Identity",
        "predictions_dir": str(pred_root),
        "dataset_root": str(ds_root),
        "copied": copied,
        "missing": missing,
        "coverage": len(copied) / max(len(samples), 1),
    }


def materialize_noise_baseline(
    predictions_dir: str | Path,
    samples: list[MMAESample],
    *,
    dataset_root: str | Path,
    snr_db: float = 8.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Write noise-corrupted input1 to ``output.wav`` (paper Noise baseline proxy)."""
    pred_root = Path(predictions_dir)
    pred_root.mkdir(parents=True, exist_ok=True)
    ds_root = Path(dataset_root)
    copied: list[str] = []
    missing: list[str] = []
    for i, sample in enumerate(samples):
        if not sample.audio_paths:
            missing.append(sample.sample_id)
            continue
        src = ds_root / sample.audio_paths[0]
        if not src.is_file():
            missing.append(sample.sample_id)
            continue
        out_dir = pred_root / sample.sample_id
        dst = out_dir / "output.wav"
        _write_noisy_wav(
            src,
            dst,
            snr_db=snr_db,
            seed=_sample_seed(sample.sample_id, seed + i),
        )
        copied.append(sample.sample_id)
    return {
        "baseline": "Noise",
        "snr_db": snr_db,
        "predictions_dir": str(pred_root),
        "dataset_root": str(ds_root),
        "copied": copied,
        "missing": missing,
        "coverage": len(copied) / max(len(samples), 1),
    }


def materialize_baseline(
    name: str,
    predictions_dir: str | Path,
    samples: list[MMAESample],
    *,
    dataset_root: str | Path,
    seed: int = 0,
    snr_db: float = 8.0,
) -> dict[str, Any]:
    if name == "Identity":
        return materialize_identity_baseline(predictions_dir, samples, dataset_root=dataset_root)
    if name == "Noise":
        return materialize_noise_baseline(
            predictions_dir,
            samples,
            dataset_root=dataset_root,
            snr_db=snr_db,
            seed=seed,
        )
    raise ValueError(f"no local baseline builder for {name!r}; use external editor inference")


def resolve_editor_command(model: str) -> str | None:
    """Expand env-based upstream editor command template for *model*."""
    template = EDITOR_ENV_COMMANDS.get(model)
    if template is None:
        return None
    return os.path.expandvars(template)


def editor_run_plan(
    model: str,
    *,
    dataset_root: str | Path,
    predictions_dir: str | Path,
    meta_json: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """CLI-friendly plan to run an upstream instruction editor (not executed here)."""
    info = EDITOR_REGISTRY.get(model)
    if info is None:
        raise ValueError(f"unknown editor model: {model!r}")
    root = Path(dataset_root)
    pred = Path(predictions_dir)
    meta = Path(meta_json) if meta_json else root / "meta" / "MMAE-meta.json"
    cmd = resolve_editor_command(model)
    return {
        "model": model,
        "kind": info.get("kind"),
        "submission": info.get("submission"),
        "dataset_root": str(root),
        "predictions_dir": str(pred),
        "meta_json": str(meta),
        "limit": limit,
        "command_template": cmd,
        "env_vars": {k: v for k, v in EDITOR_ENV_COMMANDS.items() if k == model},
        "steps": [
            f"sync-hub-audio --root {root}" + (f" --limit {limit}" if limit else ""),
            f"Run upstream editor: {cmd or 'set GOPEX_MMAE_*_CMD'}",
            f"--input {meta} --audio-root {root} --output-dir {pred}",
            f"build-predictions-json {pred} {pred.parent / f'{model.replace(' ', '_')}_predictions.json'} --root {root}",
            f"score-full {pred} {pred.parent / 'vote_cache'} --root {root} --model {model} --judger-mode omni",
        ],
        "note": info.get("note"),
    }


def materialize_editor_submission(
    model: str,
    predictions_dir: str | Path,
    samples: list[MMAESample],
    *,
    dataset_root: str | Path,
    seed: int = 0,
    snr_db: float = 8.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Materialize local baselines or return an upstream editor run plan."""
    info = EDITOR_REGISTRY.get(model)
    if info is None:
        raise ValueError(f"unknown editor model: {model!r}")
    if info.get("submission") == "local":
        if dry_run:
            return {"model": model, "dry_run": True, "action": "materialize_baseline", "builder": info.get("builder")}
        return materialize_baseline(
            model,
            predictions_dir,
            samples,
            dataset_root=dataset_root,
            seed=seed,
            snr_db=snr_db,
        )
    plan = editor_run_plan(
        model,
        dataset_root=dataset_root,
        predictions_dir=predictions_dir,
        limit=len(samples),
    )
    if dry_run:
        plan["dry_run"] = True
        return plan
    raise RuntimeError(
        f"editor {model!r} requires upstream inference; set GOPEX_MMAE_*_CMD and run plan steps"
    )
