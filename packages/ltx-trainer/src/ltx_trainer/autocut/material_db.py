"""Material database and FAISS-style nearest-neighbor retrieval (Sec. 3.6, Supp. 7.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ltx_trainer.autocut.benchmark import AdCase, BENCHMARK_CASES
from ltx_trainer.autocut.config import AutoCutConfig
from ltx_trainer.autocut.encoders import clip_embedding_from_frames, encode_audio_segment, encode_visual_frames
from ltx_trainer.autocut.rendering import RetrievedClip, faiss_nearest


@dataclass
class VideoMaterial:
    photo_id: str
    frame_id: str
    clip_id: str
    embedding: np.ndarray
    start_frame: int
    duration_frames: int
    asr_line: str = ""


@dataclass
class AudioMaterial:
    audio_id: str
    embedding: np.ndarray
    description: str = ""


@dataclass
class MaterialDatabase:
    """In-memory video/audio asset store with brute-force NN search."""

    videos: list[VideoMaterial] = field(default_factory=list)
    audios: list[AudioMaterial] = field(default_factory=list)

    def video_matrix(self) -> np.ndarray:
        if not self.videos:
            return np.zeros((0, 128), dtype=np.float64)
        return np.stack([v.embedding.reshape(-1) for v in self.videos], axis=0)

    def audio_matrix(self) -> np.ndarray:
        if not self.audios:
            return np.zeros((0, 2048), dtype=np.float64)
        return np.stack([a.embedding.reshape(-1) for a in self.audios], axis=0)

    def search_video(self, query: np.ndarray, *, top_k: int = 1) -> list[VideoMaterial]:
        mat = self.video_matrix()
        if mat.shape[0] == 0:
            return []
        idxs = faiss_nearest(query, mat, top_k=top_k)
        return [self.videos[i] for i in idxs]

    def search_audio(self, query: np.ndarray, *, top_k: int = 1) -> list[AudioMaterial]:
        mat = self.audio_matrix()
        if mat.shape[0] == 0:
            return []
        idxs = faiss_nearest(query, mat, top_k=top_k)
        return [self.audios[i] for i in idxs]

    def retrieve_clips_from_tokens(
        self,
        token_embeddings: Sequence[np.ndarray],
        *,
        strategy: str = "by_clip",
    ) -> list[RetrievedClip]:
        clips: list[RetrievedClip] = []
        for emb in token_embeddings:
            hit = self.search_video(np.asarray(emb), top_k=1)
            if not hit:
                continue
            v = hit[0]
            clips.append(
                RetrievedClip(
                    frame_id=v.frame_id,
                    clip_id=v.clip_id,
                    start_frame=v.start_frame,
                    duration_frames=v.duration_frames,
                )
            )
        return clips

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_assets": len(self.videos),
            "audio_assets": len(self.audios),
        }


def _parse_clip_timestamp(ts: str, *, default_start: int = 0, default_dur: int = 24) -> tuple[int, int]:
    """Parse '[Clip N]: Xs ~ Ys' into start/duration at 1 fps."""
    import re

    m = re.search(r"(\d+)\s*s\s*~\s*(\d+)\s*s", ts)
    if not m:
        return default_start, default_dur
    start = int(m.group(1))
    end = int(m.group(2))
    return start, max(1, end - start)


def build_database_from_case(
    case: AdCase,
    cfg: AutoCutConfig | None = None,
    *,
    seed: int = 0,
    pool_size: int = 16,
) -> MaterialDatabase:
    """Synthetic material DB aligned with benchmark case scripts."""
    c = cfg or AutoCutConfig()
    rng = np.random.default_rng(seed)
    db = MaterialDatabase()
    photo = case.case_id.replace("_", "")[:8]

    for i in range(pool_size):
        frames = encode_visual_frames(8, c, seed=seed + i)
        emb = clip_embedding_from_frames(frames)
        start, dur = _parse_clip_timestamp(
            case.clip_timestamps[i % len(case.clip_timestamps)] if case.clip_timestamps else "",
            default_start=i,
            default_dur=24,
        )
        line = case.script_lines[i % len(case.script_lines)] if case.script_lines else ""
        db.videos.append(
            VideoMaterial(
                photo_id=photo,
                frame_id=f"{photo}{start:04d}",
                clip_id=f"{photo}{start:04d}{dur:03d}",
                embedding=emb.astype(np.float64),
                start_frame=start,
                duration_frames=dur,
                asr_line=line,
            )
        )

    for j in range(8):
        db.audios.append(
            AudioMaterial(
                audio_id=f"bgm_{photo}_{j:02d}",
                embedding=encode_audio_segment(30.0 + j, c, seed=seed + 100 + j).astype(np.float64),
                description=f"{case.product_type} upbeat moderate tempo electronic",
            )
        )
    # Perturb negatives so retrieval is not uniform
    for v in db.videos:
        v.embedding += rng.normal(scale=0.01, size=v.embedding.shape)
    return db


def build_demo_database(cfg: AutoCutConfig | None = None, *, seed: int = 0) -> MaterialDatabase:
    """Merge material from all benchmark cases."""
    c = cfg or AutoCutConfig()
    merged = MaterialDatabase()
    for i, case in enumerate(BENCHMARK_CASES):
        part = build_database_from_case(case, c, seed=seed + i * 17)
        merged.videos.extend(part.videos)
        merged.audios.extend(part.audios)
    return merged
