"""Resolve inference defaults from GOPEX / Kino environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _first_existing(*paths: str | Path | None) -> Path | None:
    for p in paths:
        if not p:
            continue
        path = Path(p).expanduser()
        if path.is_file() or path.is_dir():
            return path
    return None


def _latest_lora(run_dir: Path) -> Path | None:
    ck = run_dir / "checkpoints"
    if not ck.is_dir():
        return None
    matches = sorted(ck.glob("lora_weights_step_*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


@dataclass
class InferEnv:
    """Checkpoint, LoRA, and delivery knobs for LTX3 minute renders."""

    work_root: Path
    native_ltx: Path
    gemma: Path
    lora_ckpt: Path | None
    phase2_run: Path
    distilled_lora: Path | None
    spatial_upsampler: Path | None
    temporal_upsampler: Path | None
    text_stack: Path | None
    bridge_rank: int
    height: int = 576
    width: int = 1024
    fps: float = 24.0
    inference_steps: int = 30
    guidance_scale: float = 4.0
    with_audio: bool = False
    hq: bool = False
    hq_frames_cap: int = 121
    hq_steps: int = 15
    trainer_dir: Path = field(default_factory=lambda: Path("kino/packages/ltx-trainer"))
    repo_root: Path = field(default_factory=lambda: Path("."))

    @classmethod
    def from_env(cls, *, repo_root: str | Path | None = None) -> InferEnv:
        repo = Path(repo_root or os.environ.get("GOPEX_REPO", ".")).resolve()
        work = Path(os.environ.get("WORK_ROOT", "/run/media/tim/Expansion/gopex-ltx/gemma31b-r512")).expanduser()
        phase2 = Path(
            os.environ.get(
                "GOPEX_NATIVE_PHASE2_RUN",
                os.environ.get("PHASE2_NATIVE_RUN", str(work / "runs/phase2_native_gemma31b_lora128_plus_gphotos")),
            )
        ).expanduser()
        native = Path(
            os.environ.get("NATIVE_LTX", str(work / "models/ltx-2.3-22b-gemma4-native"))
        ).expanduser()
        gemma = Path(os.environ.get("GEMMA", str(repo / "models/gemma4_31b_it"))).expanduser()
        lora_env = os.environ.get("LORA_CKPT", "").strip()
        lora = _first_existing(lora_env, _latest_lora(phase2))
        home = Path.home()
        distilled = _first_existing(
            os.environ.get("DISTILLED_LORA"),
        )
        spatial = _first_existing(
            os.environ.get("UPSAMPLER"),
        )
        temporal = _first_existing(
            os.environ.get("GOPEX_TEMPORAL_UPSAMPLER"),
        )
        try:
            from gopex_ltx.native_assets import resolve_native_assets

            assets = resolve_native_assets(repo_root=repo)
            distilled = distilled or assets.distilled_lora
            spatial = spatial or assets.spatial_upsampler
            temporal = temporal or assets.temporal_upsampler
        except ImportError:
            distilled = distilled or _first_existing(
                home / "ComfyUI/models/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            )
            spatial = spatial or _first_existing(
                home / "ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            )
            temporal = temporal or _first_existing(
                home / "ComfyUI/models/latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
            )
        text_stack = _first_existing(
            os.environ.get("GOPEX_TEXT_STACK"),
            work / "precomputed/homemovies/text_stack_weights_preprocess.safetensors",
        )
        bridge = int(os.environ.get("BRIDGE_RANK", os.environ.get("GOPEX_BRIDGE_RANK", "512")))
        with_audio = os.environ.get("GOPEX_DELIVERY_WITH_AUDIO", "0").strip().lower() not in ("0", "false", "no")
        hq = os.environ.get("GOPEX_LTX3_HQ", os.environ.get("GOPEX_DELIVERY_HQ", "0")).strip() in ("1", "true", "yes")
        return cls(
            work_root=work,
            native_ltx=native,
            gemma=gemma,
            lora_ckpt=lora,
            phase2_run=phase2,
            distilled_lora=distilled,
            spatial_upsampler=spatial,
            temporal_upsampler=temporal,
            text_stack=text_stack,
            bridge_rank=bridge,
            height=int(os.environ.get("GOPEX_LTX3_HEIGHT", "576")),
            width=int(os.environ.get("GOPEX_LTX3_WIDTH", "1024")),
            fps=float(os.environ.get("GOPEX_DELIVERY_FPS", "24")),
            inference_steps=int(os.environ.get("GOPEX_LTX3_STEPS", "30")),
            guidance_scale=float(os.environ.get("GOPEX_LTX3_GUIDANCE", "4.0")),
            with_audio=with_audio,
            hq=hq,
            hq_frames_cap=int(os.environ.get("GOPEX_DELIVERY_HQ_FRAMES", "121")),
            hq_steps=int(os.environ.get("GOPEX_DELIVERY_HQ_STEPS", "15")),
            trainer_dir=repo / "kino/packages/ltx-trainer",
            repo_root=repo,
        )

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.native_ltx.exists():
            issues.append(f"missing native LTX: {self.native_ltx}")
        if not self.gemma.is_dir():
            issues.append(f"missing Gemma root: {self.gemma}")
        if self.lora_ckpt is None or not self.lora_ckpt.is_file():
            issues.append("missing LORA_CKPT (set env or complete phase2)")
        if self.hq:
            if self.distilled_lora is None:
                issues.append("HQ mode: missing distilled LoRA")
            if self.spatial_upsampler is None:
                issues.append("HQ mode: missing spatial upsampler")
        return issues

    def to_dict(self) -> dict:
        return {
            "work_root": str(self.work_root),
            "native_ltx": str(self.native_ltx),
            "gemma": str(self.gemma),
            "lora_ckpt": str(self.lora_ckpt) if self.lora_ckpt else None,
            "phase2_run": str(self.phase2_run),
            "hq": self.hq,
            "with_audio": self.with_audio,
            "temporal_upsampler": str(self.temporal_upsampler) if self.temporal_upsampler else None,
            "height": self.height,
            "width": self.width,
            "fps": self.fps,
        }
