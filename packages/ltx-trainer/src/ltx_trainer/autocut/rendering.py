"""Token decode → FAISS retrieval → ffmpeg render (Sec. 3.6, Supp. 7.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ltx_trainer.autocut.taxonomy import RenderStrategy


@dataclass
class RetrievedClip:
    frame_id: str
    clip_id: str
    start_frame: int
    duration_frames: int


@dataclass
class RenderPlan:
    clips: list[RetrievedClip]
    script_lines: list[str]
    bgm_audio_id: str | None
    strategy: RenderStrategy
    tts_voice: str = "avatar_default"
    subtitle_style: str = "bottom_center"

    def to_dict(self) -> dict[str, Any]:
        return {
            "clips": [c.__dict__ for c in self.clips],
            "script_lines": self.script_lines,
            "bgm_audio_id": self.bgm_audio_id,
            "strategy": self.strategy.value,
            "tts_voice": self.tts_voice,
            "subtitle_style": self.subtitle_style,
        }


def parse_frame_id(frame_id: str) -> tuple[str, int]:
    """photo_id + 4-digit frame index."""
    if len(frame_id) < 5:
        return frame_id, 0
    return frame_id[:-4], int(frame_id[-4:])


def parse_clip_id(clip_id: str) -> tuple[str, int, int]:
    """photo_id + 4-digit start + 3-digit duration (1 fps)."""
    if len(clip_id) < 8:
        return clip_id, 0, 1
    return clip_id[:-7], int(clip_id[-7:-3]), int(clip_id[-3:])


def faiss_nearest(
    query: np.ndarray,
    index_embeddings: np.ndarray,
    *,
    top_k: int = 1,
) -> list[int]:
    """Brute-force cosine NN (drop-in for FAISS IndexFlatL2 / IP)."""
    q = query.astype(np.float64).reshape(1, -1)
    db = index_embeddings.astype(np.float64)
    if db.ndim == 1:
        db = db.reshape(1, -1)
    if db.shape[0] == 0:
        return []
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    dbn = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)
    sims = (dbn @ qn.T).reshape(-1)
    order = np.argsort(-sims)[:top_k]
    return [int(i) for i in order]


def build_render_plan(
    *,
    video_token_embeddings: Sequence[np.ndarray],
    script_lines: Sequence[str],
    bgm_audio_id: str | None = None,
    strategy: RenderStrategy = RenderStrategy.BY_CLIP,
    material_index: np.ndarray | None = None,
) -> RenderPlan:
    rng = np.random.default_rng(0)
    index = material_index if material_index is not None else rng.normal(size=(16, 128))
    clips: list[RetrievedClip] = []
    for i, emb in enumerate(video_token_embeddings):
        nn = faiss_nearest(np.asarray(emb), index, top_k=1)
        photo = f"ad{nn[0]:04d}" if nn else f"ad{i:04d}"
        if strategy == RenderStrategy.BY_FRAME:
            start = i * 24
            dur = 24
        else:
            start = i * 24
            dur = 24
        clips.append(
            RetrievedClip(
                frame_id=f"{photo}{start:04d}",
                clip_id=f"{photo}{start:04d}{dur:03d}",
                start_frame=start,
                duration_frames=dur,
            )
        )
    return RenderPlan(
        clips=clips,
        script_lines=list(script_lines),
        bgm_audio_id=bgm_audio_id,
        strategy=strategy,
    )


def ffmpeg_render_command(plan: RenderPlan, output_path: str) -> list[str]:
    """ffmpeg argv: concat clips, burn subtitles, mix TTS + BGM (dry-run paths)."""
    inputs = [f"clip_{i}.mp4" for i in range(len(plan.clips))]
    cmd = ["ffmpeg", "-y"]
    for inp in inputs:
        cmd.extend(["-i", inp])
    if plan.bgm_audio_id:
        cmd.extend(["-i", f"{plan.bgm_audio_id}.wav"])
    n = len(inputs)
    subs = []
    for i, line in enumerate(plan.script_lines[:n]):
        safe = line.replace("'", "\\'").replace(":", "\\:")
        subs.append(f"drawtext=text='{safe}':x=(w-text_w)/2:y=h-80:fontsize=24:fontcolor=white")
    vf = f"concat=n={n}:v=1:a=0"
    if subs:
        vf = vf + "," + ",".join(subs)
    cmd.extend(["-filter_complex", vf, "-map", "[outv]"])
    if plan.bgm_audio_id:
        cmd.extend(
            [
                "-filter_complex",
                f"[{n}:a]volume=0.35[bgm];[0:a][bgm]amix=inputs=2:duration=first[aout]",
                "-map",
                "[aout]",
            ]
        )
    cmd.append(output_path)
    return cmd


def render_pipeline_steps(plan: RenderPlan) -> list[str]:
    """Human-readable post-processing checklist (Supp. 7.2)."""
    return [
        "decode video tokens → embeddings",
        "FAISS NN → (frame_id, clip_id)",
        f"assemble via {plan.strategy.value}",
        "overlay subtitles per script line",
        "TTS voice-over per line",
        f"mix BGM track {plan.bgm_audio_id or 'none'}",
        "export MP4",
    ]
