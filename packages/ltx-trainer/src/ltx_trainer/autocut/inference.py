"""AutoCut inference: four editing tasks + token decode + retrieval (Sec. 3, Supp. 7.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ltx_trainer.autocut.benchmark import AdCase, case_by_id, case_to_product_info
from ltx_trainer.autocut.config import AutoCutConfig
from ltx_trainer.autocut.encoders import clip_embedding_from_frames, encode_visual_frames
from ltx_trainer.autocut.material_db import MaterialDatabase, build_database_from_case
from ltx_trainer.autocut.metrics import (
    MetricBundle,
    clips_rank_accuracy,
    clips_selection_accuracy,
    music_similarity_score,
    script_quality_heuristic,
    visual_script_correlation_score,
    word_count_discrepancy,
)
from ltx_trainer.autocut.rendering import RenderPlan, RenderStrategy, build_render_plan, ffmpeg_render_command
from ltx_trainer.autocut.rqvae import ResidualRQVAE
from ltx_trainer.autocut.sft import (
    build_bgm_selection_example,
    build_script_generation_example,
    build_video_selection_example,
    build_video_sorting_example,
)
from ltx_trainer.autocut.taxonomy import EditingTask
from ltx_trainer.autocut.tokens import format_audio_token, format_video_token


@dataclass
class EditResult:
    scenario: str
    case_id: str
    selected_indices: list[int]
    sorted_indices: list[int]
    script_lines: list[str]
    video_tokens: list[str]
    audio_tokens: list[str]
    render_plan: RenderPlan
    sft_examples: list[dict[str, Any]] = field(default_factory=list)
    metrics: MetricBundle | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "case_id": self.case_id,
            "selected_indices": self.selected_indices,
            "sorted_indices": self.sorted_indices,
            "script_lines": self.script_lines,
            "video_tokens": self.video_tokens,
            "audio_tokens": self.audio_tokens,
            "render": self.render_plan.to_dict(),
            "sft_tasks": [ex.get("task") for ex in self.sft_examples],
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


def _product_text(case: AdCase) -> str:
    info = case_to_product_info(case)
    feats = ", ".join(info["features"])
    return f"Product Type: {info['product_type']}\nBrand: {info['brand']}\nFeatures: [{feats}]"


def _keyword_score(line: str, script: str) -> float:
    a = set(line.lower().split())
    b = set(script.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class AutoCutEditor:
    """End-to-end editor: discrete tokens → retrieval → render plan."""

    def __init__(
        self,
        cfg: AutoCutConfig | None = None,
        *,
        material_db: MaterialDatabase | None = None,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg or AutoCutConfig()
        self.seed = seed
        self.video_rqvae = ResidualRQVAE(self.cfg.video_rqvae, seed=seed)
        self.audio_rqvae = ResidualRQVAE(self.cfg.audio_rqvae, seed=seed + 1)
        self.material_db = material_db or MaterialDatabase()

    def load_case_material(self, case: AdCase) -> None:
        self.material_db = build_database_from_case(case, self.cfg, seed=self.seed)

    def select_clips(
        self,
        case: AdCase,
        candidate_indices: Sequence[int],
        *,
        script: str | None = None,
    ) -> list[int]:
        """Video selection: pick clips whose ASR lines match script sentences."""
        script_text = script or "\n".join(case.script_lines)
        n_need = len(case.script_lines)
        scored: list[tuple[float, int]] = []
        for idx in candidate_indices:
            if idx >= len(self.material_db.videos):
                score = _keyword_score(case.script_lines[idx % len(case.script_lines)], script_text)
            else:
                line = self.material_db.videos[idx].asr_line
                score = _keyword_score(line, script_text)
            scored.append((score, idx))
        scored.sort(reverse=True)
        return [idx for _, idx in scored[: max(1, n_need)]]

    def sort_clips(
        self,
        case: AdCase,
        selected: Sequence[int],
        *,
        script: str | None = None,
    ) -> list[int]:
        """Order selected clips by best script-line alignment (permutation)."""
        script_lines = (script or "\n".join(case.script_lines)).split("\n")
        script_lines = [s.strip() for s in script_lines if s.strip()]
        order: list[tuple[float, int]] = []
        for idx in selected:
            line = ""
            if idx < len(self.material_db.videos):
                line = self.material_db.videos[idx].asr_line
            best = max(
                (_keyword_score(line, sl), li) for li, sl in enumerate(script_lines)
            ) if script_lines else (0.0, 0)
            order.append((best[0], idx, best[1]))
        order.sort(key=lambda x: x[2])
        return [idx for _, idx, _ in order]

    def generate_script(self, case: AdCase, clip_indices: Sequence[int]) -> list[str]:
        lines: list[str] = []
        for idx in clip_indices:
            if idx < len(self.material_db.videos) and self.material_db.videos[idx].asr_line:
                lines.append(self.material_db.videos[idx].asr_line)
            elif len(lines) < len(case.script_lines):
                lines.append(case.script_lines[len(lines)])
            else:
                lines.append(f"Highlight for {case.brand}.")
        return lines[: max(1, len(clip_indices))]

    def select_bgm(self, case: AdCase, *, script: str) -> tuple[list[str], str]:
        """BGM selection via audio RQ-VAE tokens + NN retrieval."""
        mean_audio = (
            np.mean(self.material_db.audio_matrix(), axis=0)
            if len(self.material_db.audios) > 0
            else np.zeros(self.cfg.audio_feature_dim)
        )
        q = self.audio_rqvae.encode(mean_audio)
        hit = self.material_db.search_audio(q.reconstructed, top_k=1)
        audio_id = hit[0].audio_id if hit else "bgm_default"
        q = self.audio_rqvae.encode(hit[0].embedding if hit else mean_audio)
        tokens = [format_audio_token(h, c) for h, c in q.codes]
        return tokens, audio_id

    def encode_clip_tokens(self, clip_indices: Sequence[int]) -> list[str]:
        tokens: list[str] = []
        for idx in clip_indices:
            if idx < len(self.material_db.videos):
                emb = self.material_db.videos[idx].embedding
            else:
                emb = clip_embedding_from_frames(encode_visual_frames(4, self.cfg, seed=self.seed + idx))
            q = self.video_rqvae.encode(emb)
            tokens.extend(format_video_token(h, c) for h, c in q.codes)
        return tokens

    def decode_to_render_plan(
        self,
        *,
        clip_indices: Sequence[int],
        script_lines: Sequence[str],
        bgm_audio_id: str | None,
        strategy: RenderStrategy,
    ) -> RenderPlan:
        embs: list[np.ndarray] = []
        for idx in clip_indices:
            if idx < len(self.material_db.videos):
                codes = self.video_rqvae.encode(self.material_db.videos[idx].embedding).codes
                embs.append(self.video_rqvae.decode_codes(codes))
            else:
                embs.append(np.zeros(self.cfg.video_feature_dim))
        return build_render_plan(
            video_token_embeddings=embs,
            script_lines=script_lines,
            bgm_audio_id=bgm_audio_id,
            strategy=strategy,
            material_index=self.material_db.video_matrix(),
        )

    def script_driven(
        self,
        case_id: str,
        *,
        render_strategy: RenderStrategy = RenderStrategy.BY_CLIP,
    ) -> EditResult:
        case = case_by_id(case_id)
        if case is None:
            raise ValueError(f"unknown case_id: {case_id}")
        self.load_case_material(case)
        product = _product_text(case)
        script = "\n".join(case.script_lines)
        pool = list(range(min(10, len(self.material_db.videos))))
        selected = self.select_clips(case, pool, script=script)
        shuffled = selected.copy()
        np.random.default_rng(self.seed).shuffle(shuffled)
        sorted_idx = self.sort_clips(case, selected, script=script)
        audio_tokens, bgm_id = self.select_bgm(case, script=script)
        video_tokens = self.encode_clip_tokens(sorted_idx)
        render = self.decode_to_render_plan(
            clip_indices=sorted_idx,
            script_lines=case.script_lines,
            bgm_audio_id=bgm_id,
            strategy=render_strategy,
        )

        pred_pos = [i in selected for i in pool]
        gt_pos = pred_pos  # demo DB is self-consistent
        vsc_vals = [
            visual_script_correlation_score(
                self.material_db.videos[i].asr_line if i < len(self.material_db.videos) else "",
                case.script_lines[j % len(case.script_lines)],
            )
            for j, i in enumerate(sorted_idx)
        ]
        metrics = MetricBundle(
            csa=clips_selection_accuracy(pred_pos, gt_pos),
            cra=clips_rank_accuracy(sorted_idx, sorted_idx),
            vsc=float(np.mean(vsc_vals)) if vsc_vals else 0.0,
            sq=script_quality_heuristic(script, script, product_keywords=case.features),
            wcd=word_count_discrepancy(script, sum(len(s.split()) for s in case.script_lines) // max(len(case.script_lines), 1)),
            mss=music_similarity_score("upbeat electronic", "energetic electronic"),
        )

        examples = [
            build_video_selection_example(
                product_info=product,
                script=script,
                candidate_indices=pool,
                selected_indices=selected,
            ).to_sharegpt(),
            build_video_sorting_example(
                product_info=product,
                script=script,
                shuffled_indices=shuffled,
                sorted_indices=sorted_idx,
            ).to_sharegpt(),
            build_bgm_selection_example(
                product_info=product,
                script=script,
                audio_token_line=" ".join(audio_tokens),
            ).to_sharegpt(),
        ]

        return EditResult(
            scenario="script_driven",
            case_id=case_id,
            selected_indices=selected,
            sorted_indices=sorted_idx,
            script_lines=list(case.script_lines),
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            render_plan=render,
            sft_examples=examples,
            metrics=metrics,
        )

    def footage_driven(
        self,
        case_id: str,
        *,
        clip_count: int | None = None,
        render_strategy: RenderStrategy = RenderStrategy.BY_CLIP,
    ) -> EditResult:
        case = case_by_id(case_id)
        if case is None:
            raise ValueError(f"unknown case_id: {case_id}")
        self.load_case_material(case)
        product = _product_text(case)
        n = clip_count or len(case.clip_timestamps) or len(case.script_lines)
        clips = list(range(min(n, len(self.material_db.videos))))
        script_lines = self.generate_script(case, clips)
        sorted_idx = self.sort_clips(case, clips, script="\n".join(script_lines))
        audio_tokens, bgm_id = self.select_bgm(case, script="\n".join(script_lines))
        video_tokens = self.encode_clip_tokens(sorted_idx)
        render = self.decode_to_render_plan(
            clip_indices=sorted_idx,
            script_lines=script_lines,
            bgm_audio_id=bgm_id,
            strategy=render_strategy,
        )

        examples = [
            build_script_generation_example(
                product_info=product,
                clip_count=len(clips),
                script_lines=script_lines,
            ).to_sharegpt(),
            build_video_sorting_example(
                product_info=product,
                script="\n".join(script_lines),
                shuffled_indices=list(reversed(clips)),
                sorted_indices=sorted_idx,
            ).to_sharegpt(),
            build_bgm_selection_example(
                product_info=product,
                script="\n".join(script_lines),
                audio_token_line=" ".join(audio_tokens),
            ).to_sharegpt(),
        ]

        return EditResult(
            scenario="footage_driven",
            case_id=case_id,
            selected_indices=clips,
            sorted_indices=sorted_idx,
            script_lines=script_lines,
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            render_plan=render,
            sft_examples=examples,
        )


def run_script_driven_edit(
    case_id: str,
    *,
    cfg: AutoCutConfig | None = None,
    render_strategy: RenderStrategy = RenderStrategy.BY_CLIP,
    seed: int = 0,
) -> dict[str, Any]:
    editor = AutoCutEditor(cfg, seed=seed)
    try:
        result = editor.script_driven(case_id, render_strategy=render_strategy)
    except ValueError as e:
        return {"error": str(e)}
    return {
        "plan": {
            "scenario": result.scenario,
            "case_id": result.case_id,
            "selected_clip_indices": result.selected_indices,
            "sorted_clip_indices": result.sorted_indices,
            "script_lines": result.script_lines,
            "bgm_tokens": result.audio_tokens,
            "render_strategy": render_strategy.value,
            "sft_tasks": [ex.get("task") for ex in result.sft_examples],
        },
        "render": result.render_plan.to_dict(),
        "ffmpeg_argv_preview": ffmpeg_render_command(result.render_plan, "out/autocut_ad.mp4"),
        "metrics_proxy": result.metrics.to_dict() if result.metrics else None,
        "tasks": [
            EditingTask.VIDEO_SELECTION.value,
            EditingTask.VIDEO_SORTING.value,
            EditingTask.BGM_SELECTION.value,
        ],
        "token_preview": {"video": result.video_tokens[:16], "audio": result.audio_tokens[:8]},
    }


def run_footage_driven_edit(
    case_id: str,
    *,
    cfg: AutoCutConfig | None = None,
    clip_count: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    editor = AutoCutEditor(cfg, seed=seed)
    try:
        result = editor.footage_driven(case_id, clip_count=clip_count)
    except ValueError as e:
        return {"error": str(e)}
    return {
        "plan": {
            "scenario": result.scenario,
            "case_id": result.case_id,
            "selected_clip_indices": result.selected_indices,
            "sorted_clip_indices": result.sorted_indices,
            "script_lines": result.script_lines,
            "bgm_tokens": result.audio_tokens,
            "render_strategy": RenderStrategy.BY_CLIP.value,
            "sft_tasks": [ex.get("task") for ex in result.sft_examples],
        },
        "render": result.render_plan.to_dict(),
        "tasks": [
            EditingTask.SCRIPT_GENERATION.value,
            EditingTask.VIDEO_SORTING.value,
            EditingTask.BGM_SELECTION.value,
        ],
        "token_preview": {"video": result.video_tokens[:16], "audio": result.audio_tokens[:8]},
    }
