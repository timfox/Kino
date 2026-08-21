"""Framework card, demos, benchmark manifest (arXiv:2605.30170)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.vlm_counting.config import VLMCountingConfig
from ltx_trainer.vlm_counting.gaps import (
    CountingDiagnostics,
    comparative_match_accuracy,
    regime_for_n,
    synthetic_predicted_count,
)
from ltx_trainer.vlm_counting.go_board import patch_embeddings_from_board, sample_go_board
from ltx_trainer.vlm_counting.intervention import steering_accuracy_curve
from ltx_trainer.vlm_counting.ltx_bridge import caption_vision_gap_check, ltx_integration_notes
from ltx_trainer.vlm_counting.metrics import (
    fig2_baseline_paradox,
    fig3_gap_curves,
    fig4_comparative_counting,
    fig5_attractor_distribution,
    fig6_circuit_overlap,
    fig7_qwen_baseline,
    fig8_qwen_gaps,
    fig9_layer_probe_synthetic,
    steering_curve_d11,
    three_stages,
)
from ltx_trainer.vlm_counting.prompts import prompt_bundle
from ltx_trainer.vlm_counting.probing import StonePresenceProbe, train_probe_on_id


def framework_card(cfg: VLMCountingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VLMCountingConfig()
    return {
        "name": "Visual Counting Bottleneck",
        "paper": cfg.paper_arxiv,
        "hypothesis": "fractured_magnitude",
        "stages": three_stages(),
        "curriculum": {
            "phase1_text_max": cfg.text_pretrain_max,
            "phase2_visual_max": cfg.visual_train_max,
            "regimes": {"ID": f"0-{cfg.id_max}", "VE": f"{cfg.ve_min}-{cfg.ve_max}", "FE": f"{cfg.fe_min}-{cfg.fe_max}"},
        },
        "studies": ["toy_vlm_go_19x19", cfg.foundation_model + "_go_6x6"],
        "diagnostics": ["hidden_number_probe", "vision_gap", "language_gap", "comparative_counting"],
    }


def paper_limitations() -> list[str]:
    return [
        "Go boards isolate numerosity; real scenes add occlusion and semantic filtering.",
        "Qwen3-VL validation may not transfer to all open VLMs.",
        "Toy VLM uses 2-layer ViT/GPT stub; full retraining is external.",
    ]


def benchmark_manifest(cfg: VLMCountingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VLMCountingConfig()
    return {
        "synthetic": {"board": f"{cfg.board_size}x{cfg.board_size}", "stone_px": cfg.stone_pixels},
        "foundation": {"board": f"{cfg.qwen_board_size}x{cfg.qwen_board_size}", "model": cfg.foundation_model},
        "metrics": ["exact_match", "vision_gap", "language_gap", "comparative_accuracy", "steering_accuracy"],
        "code": "https://github.com/Russellpang/semproj",
    }


def evaluation_demo(*, cfg: VLMCountingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VLMCountingConfig()
    torch.manual_seed(11)
    probe = StonePresenceProbe(cfg.probe_dim)
    id_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for n in range(0, cfg.visual_train_max + 1, 7):
        board = sample_go_board(n, board_size=cfg.board_size, stone_pixels=cfg.stone_pixels)
        emb = patch_embeddings_from_board(board, dim=cfg.probe_dim, seed=n)
        id_samples.append((emb, board.patch_labels))
    probe_loss = train_probe_on_id(probe, id_samples, steps=30)

    extrapolation: list[dict[str, Any]] = []
    for n in [25, 55, 75, 110]:
        board = sample_go_board(n, board_size=cfg.board_size, stone_pixels=cfg.stone_pixels)
        emb = patch_embeddings_from_board(board, dim=cfg.probe_dim, seed=n + 100)
        nh = float(probe.hidden_number(emb.unsqueeze(0)).item())
        np_ = synthetic_predicted_count(n, visual_train_max=cfg.visual_train_max)
        diag = CountingDiagnostics(ng=n, nh=nh, np_=np_)
        extrapolation.append(
            {
                "n": n,
                "regime": regime_for_n(n, id_max=cfg.id_max, ve_max=cfg.ve_max),
                "nh": round(nh, 2),
                "np": np_,
                "vision_gap": round(diag.vision_gap, 2),
                "language_gap": round(diag.language_gap, 2),
                "compare_acc": round(comparative_match_accuracy(n, visual_train_max=cfg.visual_train_max), 3),
            }
        )

    ve_language_gaps = [r["language_gap"] for r in extrapolation if r["regime"] == "VE"]
    board12 = sample_go_board(12, board_size=cfg.board_size, stone_pixels=cfg.stone_pixels)
    emb12 = patch_embeddings_from_board(board12, dim=cfg.probe_dim)
    train_probe_on_id(probe, [(emb12, board12.patch_labels)], steps=20)
    steering = steering_accuracy_curve(probe, emb12, board12.black_count)
    caption_qa = caption_vision_gap_check("12 people walking", probe, emb12)

    return {
        "probe_train_loss": round(probe_loss, 4),
        "extrapolation_diagnostics": extrapolation,
        "ve_mean_language_gap": round(sum(ve_language_gaps) / max(len(ve_language_gaps), 1), 2),
        "steering_intervention": steering,
        "caption_qa_demo": caption_qa,
        "prompt_vision": prompt_bundle("vision"),
        "circuit_overlap": fig6_circuit_overlap(),
        "steering_curve_paper": steering_curve_d11(),
        "ltx_integration": ltx_integration_notes(),
        "conclusion": "Bottleneck at symbolic mapping (stage 3); vision probe stays accurate in VE.",
    }


def training_step_demo(*, cfg: VLMCountingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VLMCountingConfig()
    probe = StonePresenceProbe(cfg.probe_dim)
    board = sample_go_board(12, board_size=cfg.board_size, stone_pixels=cfg.stone_pixels)
    emb = patch_embeddings_from_board(board, dim=cfg.probe_dim)
    loss = train_probe_on_id(probe, [(emb, board.patch_labels)], steps=5)
    nh = float(probe.hidden_number(emb.unsqueeze(0)).item())
    return {"probe_loss": round(loss, 4), "hidden_number": round(nh, 2), "ground_truth": board.black_count}
