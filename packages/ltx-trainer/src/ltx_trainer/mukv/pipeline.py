"""MuKV framework card, paper tables, and smoke demos (arXiv:2605.22269)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mukv.compression import compress_kv_stub
from ltx_trainer.mukv.config import MuKVConfig
from ltx_trainer.mukv.layout import LIMITATIONS
from ltx_trainer.mukv.mock import toy_attention, toy_keys, toy_question_vec
from ltx_trainer.mukv.retrieval import semi_hierarchical_retrieval


def framework_card(cfg: MuKVConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MuKVConfig()
    return {
        "name": "MuKV",
        "paper": cfg.paper_arxiv,
        "idea": (
            "Streaming VideoQA via offline KV-cache memory and online retrieval. "
            "Offline: multi-grained KV caches (patch/frame/segment) with Dual-signal KV-cache Compression (DCP) "
            "guided by self-attention importance and Fourier-domain frequency. "
            "Online: semi-hierarchical retrieval (parallel per granularity + segment-guided reranking) for answer decoding."
        ),
        "backbone": cfg.model_name,
        "granularities": ["patch", "frame", "segment"],
        "dcp": {
            "signals": ["self_attention", "frequency_fft"],
            "fusion": "alpha * minmax(att) + (1-alpha) * minmax(freq)",
        },
        "retrieval": {"type": "semi_hierarchical", "rerank_score": "(1-lambda_g)*s + lambda_g*gamma"},
        "defaults": cfg.__dict__,
    }


def fig1_efficiency_accuracy_points() -> dict[str, list[dict[str, float]]]:
    """Fig. 1(a) excerpt points (QA accuracy vs visual tokens)."""
    return {
        "mukv": [{"tokens_k": 8.3, "rvsego_acc": 57.9, "rvsmovie_acc": 45.2}],
        "rekv": [{"tokens_k": 12.5, "rvsego_acc": 51.5, "rvsmovie_acc": 42.3}],
    }


def table1_main_results() -> list[dict[str, Any]]:
    """Table 1 — key rows (RVSEgo/RVSMovie, token counts, 0.5B/7B)."""
    return [
        {"method": "ReKV", "size": "0.5B", "inf_tok_k": 12.5, "mem_tok_k_10min": 59.0, "rvsego": 51.5, "rvsmovie": 42.3},
        {"method": "ReKV", "size": "7B", "inf_tok_k": 12.5, "mem_tok_k_10min": 59.0, "rvsego": 56.2, "rvsmovie": 48.2},
        {"method": "MuKV", "size": "0.5B", "inf_tok_k": 8.3, "mem_tok_k_10min": 59.0, "rvsego": 57.9, "rvsmovie": 45.2},
        {"method": "MuKV", "size": "7B", "inf_tok_k": 8.3, "mem_tok_k_10min": 59.0, "rvsego": 59.5, "rvsmovie": 48.5},
    ]


def table2_granularity_ablation() -> list[dict[str, Any]]:
    """Table 2 — granularity combinations (0.5B)."""
    return [
        {"patch": True, "frame": False, "segment": False, "mem_tok_k": 47.0, "rvsego": 51.6, "rvsmovie": 44.1},
        {"patch": False, "frame": True, "segment": False, "mem_tok_k": 39.0, "rvsego": 53.1, "rvsmovie": 45.2},
        {"patch": False, "frame": False, "segment": True, "mem_tok_k": 10.0, "rvsego": 54.9, "rvsmovie": 44.8},
        {"patch": True, "frame": True, "segment": True, "mem_tok_k": 59.0, "rvsego": 56.5, "rvsmovie": 46.0},
    ]


def table3_compression_ablation() -> list[dict[str, Any]]:
    """Table 3 — selected compression rows (MuKV/ ReKV)."""
    return [
        {"method": "MuKV", "compression": "none", "inf_tok_k": 12.5, "mem_tok_k": 177.0, "rvsego": 53.7, "rvsmovie": 44.3},
        {"method": "MuKV", "compression": "DCP(67%)", "inf_tok_k": 8.3, "mem_tok_k": 59.0, "rvsego": 57.3, "rvsmovie": 45.6},
        {"method": "ReKV", "compression": "none", "inf_tok_k": 12.5, "mem_tok_k": 59.0, "rvsego": 51.5, "rvsmovie": 42.3},
        {"method": "ReKV", "compression": "DCP(50%)", "inf_tok_k": 6.3, "mem_tok_k": 29.0, "rvsego": 56.1, "rvsmovie": 44.9},
        {"method": "ReKV", "compression": "DCP(90%)", "inf_tok_k": 1.3, "mem_tok_k": 6.0, "rvsego": 50.9, "rvsmovie": 46.8},
    ]


def table4_deployment_metrics() -> list[dict[str, Any]]:
    """Table 4 — s/Q and G/h (selected)."""
    return [
        {"method": "ReKV", "ratio": "0", "sec_per_q": 0.92, "gb_per_h": 4.00, "rvsego": 51.5},
        {"method": "MuKV", "ratio": "0", "sec_per_q": 0.72, "gb_per_h": 3.72, "rvsego": 53.7},
        {"method": "MuKV", "ratio": "2/3", "sec_per_q": 0.65, "gb_per_h": 1.23, "rvsego": 57.3},
        {"method": "MuKV", "ratio": "3/4", "sec_per_q": 0.59, "gb_per_h": 0.91, "rvsego": 54.9},
    ]


def table6_frequency_keep_high_vs_low() -> list[dict[str, Any]]:
    """Table 6 — keep high frequency vs keep low frequency (ReKV, 50%)."""
    return [
        {"method": "ReKV", "keep_low_freq": False, "keep_high_freq": True, "ratio": 0.5, "rvsego": 55.1, "rvsmovie": 43.9},
        {"method": "ReKV", "keep_low_freq": True, "keep_high_freq": False, "ratio": 0.5, "rvsego": 51.4, "rvsmovie": 41.4},
    ]


def table7_retrieval_methods() -> list[dict[str, Any]]:
    """Table 7 — retrieval method comparison (MuKV)."""
    return [
        {"method": "parallel", "rvsego": 56.5, "rvsmovie": 46.0},
        {"method": "hierarchical", "rvsego": 52.9, "rvsmovie": 43.1},
        {"method": "semi_hierarchical", "rvsego": 57.9, "rvsmovie": 45.2},
    ]


def pipeline_demo(cfg: MuKVConfig | None = None) -> dict[str, Any]:
    """Toy run: DCP compress indices + semi-hierarchical retrieval indices."""
    cfg = cfg or MuKVConfig()
    keys = toy_keys(cfg.patches_per_frame_P, dim=8, phase=0.2)
    attn = toy_attention(num_heads=4, num_tokens=cfg.patches_per_frame_P)
    dcp = compress_kv_stub(keys_p_by_d=keys, attn_h_by_p=attn, alpha=cfg.alpha_frame, rho=cfg.rho_frame)

    # Represent blocks by mean pooled keys of small slices (toy).
    q = toy_question_vec(dim=8)
    patch_blocks = [keys[i] for i in range(0, 80)]  # treat first 80 tokens as blocks
    frame_blocks = [keys[i] for i in range(80, 160)]
    segment_blocks = [keys[i] for i in range(160, 196)]
    rr = semi_hierarchical_retrieval(
        q=q,
        patch_blocks=patch_blocks,
        frame_blocks=frame_blocks,
        segment_blocks=segment_blocks,
        k2_patch=10,
        k2_frame=10,
        k2_segment=10,
        k_patch=5,
        k_frame=5,
        k_segment=5,
        lambda_g_patch=cfg.lambda_g_patch,
        lambda_g_frame=cfg.lambda_g_frame,
    )

    return {
        "dcp_keep_count": len(dcp["keep_indices"]),
        "dcp_keep_first5": list(dcp["keep_indices"])[:5],
        "dcp_scores_span": float(max(dcp["fused_scores"]) - min(dcp["fused_scores"])) if dcp["fused_scores"] else 0.0,
        "retrieval": rr,
    }


def evaluation_demo(cfg: MuKVConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MuKVConfig()
    t1 = table1_main_results()
    mukv05 = next(r for r in t1 if r["method"] == "MuKV" and r["size"] == "0.5B")
    rekv05 = next(r for r in t1 if r["method"] == "ReKV" and r["size"] == "0.5B")
    return {
        "framework": framework_card(cfg),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "delta_rvsego_mukv_vs_rekv_05b": round(float(mukv05["rvsego"]) - float(rekv05["rvsego"]), 1),
        "paper_tables": {
            "fig1_points": fig1_efficiency_accuracy_points(),
            "table1_main": t1,
            "table2_granularity": table2_granularity_ablation(),
            "table3_compression": table3_compression_ablation(),
            "table4_deployment": table4_deployment_metrics(),
            "table6_frequency": table6_frequency_keep_high_vs_low(),
            "table7_retrieval": table7_retrieval_methods(),
        },
    }

