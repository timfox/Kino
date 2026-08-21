"""AutoCut evaluation smoke."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.autocut.alignment import alignment_step
from ltx_trainer.autocut.benchmark import BENCHMARK_CASES, case_by_id
from ltx_trainer.autocut.config import AutoCutConfig
from ltx_trainer.autocut.dataset import alignment_dataset_card, dataset_statistics_summary, sft_dataset_card
from ltx_trainer.autocut.dataset_build import dataset_pipeline_summary
from ltx_trainer.autocut.encoders import encode_audio_segment, encode_multimodal_clip, encode_visual_frames
from ltx_trainer.autocut.inference import run_script_driven_edit
from ltx_trainer.autocut.material_db import build_demo_database
from ltx_trainer.autocut.metrics import (
    MetricBundle,
    clips_rank_accuracy,
    clips_selection_accuracy,
    music_similarity_score,
    script_quality_heuristic,
    visual_script_correlation_score,
    word_count_discrepancy,
)
from ltx_trainer.autocut.rendering import RenderStrategy, build_render_plan
from ltx_trainer.autocut.rqvae import ResidualRQVAE
from ltx_trainer.autocut.sft import build_video_selection_example
from ltx_trainer.autocut.tables import table1_main_results, table2_ablation_training, user_study_vs_gpt4o
from ltx_trainer.autocut.tokens import format_audio_token, format_video_token, serialize_alignment_sample
from ltx_trainer.autocut.taxonomy import EditingTask, TrainingStage
from ltx_trainer.autocut.training import full_training_pipeline


def evaluation_smoke(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    rng = np.random.default_rng(c.random_seed)

    frames = encode_visual_frames(8, c, seed=c.random_seed)
    audio = encode_audio_segment(12.0, c, seed=c.random_seed + 1)
    v_rq = ResidualRQVAE(c.video_rqvae, seed=0)
    a_rq = ResidualRQVAE(c.audio_rqvae, seed=1)
    v_loss = v_rq.forward(frames[0])
    a_loss = a_rq.forward(audio)

    v_tokens = [format_video_token(h, code) for h, code in v_rq.encode(frames[0]).codes]
    a_tokens = [format_audio_token(h, code) for h, code in a_rq.encode(audio).codes]
    sample = serialize_alignment_sample(
        script_blocks=["This T-shirt is a hit-crafted with precision."],
        video_token_rows=[[t] for t in v_tokens[:4]],
        audio_tokens=a_tokens,
    )

    emb_table = rng.normal(size=(512, 64))
    align = alignment_step(list(range(64)), emb_table, c)

    case = BENCHMARK_CASES[0]
    sel_ex = build_video_selection_example(
        product_info=str(case.brand),
        script="\n".join(case.script_lines),
        candidate_indices=list(range(10)),
        selected_indices=[1, 3, 5, 7],
    )
    pred_pos = [False, True, False, True, False, True, False, True, False, False]
    gt_pos = [False, True, False, True, False, True, False, True, True, False]
    csa = clips_selection_accuracy(pred_pos, gt_pos)
    cra = clips_rank_accuracy([3, 4, 5, 7], [3, 4, 5, 7])
    vsc = visual_script_correlation_score("earbuds wireless fit", case.script_lines[0]) / 2.0
    sq = script_quality_heuristic(
        " ".join(case.script_lines),
        " ".join(case.script_lines),
        product_keywords=case.features,
    )
    wcd = word_count_discrepancy(case.script_lines[0], 12)
    mss = music_similarity_score("upbeat electronic moderate tempo", "energetic electronic rhythm")

    plan = build_render_plan(
        video_token_embeddings=[v_rq.decode_codes(v_rq.encode(frames[i]).codes) for i in range(min(4, len(frames)))],
        script_lines=list(case.script_lines[:4]),
        bgm_audio_id="bgm_0042",
        strategy=RenderStrategy.BY_CLIP,
        material_index=build_demo_database(c).video_matrix(),
    )

    ours = [r for r in table1_main_results() if r["model"] == "AutoCut (ours)"][0]
    ablation = table2_ablation_training()[-1]
    edit = run_script_driven_edit(BENCHMARK_CASES[0].case_id, cfg=c)

    return {
        "paper": f"arXiv:{c.paper_arxiv}",
        "github": c.github_url,
        "base_llm": c.base_llm,
        "training_stages": [TrainingStage.MULTIMODAL_ALIGNMENT.value, TrainingStage.SUPERVISED_FINETUNING.value],
        "tasks": [t.value for t in EditingTask],
        "rqvae_video_cosine": v_loss.cosine_sim,
        "rqvae_audio_cosine": a_loss.cosine_sim,
        "alignment_loss": align.to_dict(),
        "serialized_sample_chars": len(sample),
        "sft_example_task": sel_ex.task.value,
        "metrics_smoke": MetricBundle(csa, cra, vsc, sq, wcd, mss).to_dict(),
        "paper_metrics": {
            "CSA": ours["CSA"],
            "CRA": ours["CRA"],
            "SQ": ours["SQ"],
            "WCD": ours["WCD"],
            "MSS": ours["MSS"],
        },
        "ablation_emb_sft_CRA": ablation["CRA"],
        "render_plan": plan.to_dict(),
        "cost_ratio_gpt4o_vs_autocut": c.gpt4o_cost_per_100_videos_usd / c.inference_cost_per_100_videos_usd,
        "user_study": user_study_vs_gpt4o(),
        "alignment_dataset": alignment_dataset_card(c),
        "sft_dataset": sft_dataset_card(c),
        "dataset_stats": dataset_statistics_summary(),
        "dataset_pipeline": dataset_pipeline_summary(c),
        "training_pipeline": full_training_pipeline(c),
        "material_db": build_demo_database(c).to_dict(),
        "multimodal_clip": {
            "fps": encode_multimodal_clip(cfg=c).fps,
            "visual_shape": list(encode_multimodal_clip(cfg=c).visual.shape),
            "audio_shape": list(encode_multimodal_clip(cfg=c).audio.shape),
        },
        "benchmark_cases": [bc.case_id for bc in BENCHMARK_CASES],
        "editing_script_driven": edit.get("plan", {}).get("scenario"),
    }


def run_case_demo(case_id: str, cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    case = case_by_id(case_id)
    if case is None:
        return {"error": f"unknown case_id {case_id}"}
    smoke = evaluation_smoke(c)
    return {
        "case_id": case.case_id,
        "product": {
            "product_type": case.product_type,
            "brand": case.brand,
            "features": list(case.features),
        },
        "script_lines": list(case.script_lines),
        "clip_timestamps": list(case.clip_timestamps),
        "pipeline": smoke,
    }
