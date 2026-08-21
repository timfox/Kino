"""ST-SFLora toy STE + alternating optimization smoke (arXiv:2605.26120)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.st_sflora.config import StSfloraConfig
from ltx_trainer.st_sflora.optimize import ClientLinkState, alternating_optimize
from ltx_trainer.st_sflora.radio import client_selected, standing_time_s
from ltx_trainer.st_sflora.ste import semantic_transmission_efficiency
from ltx_trainer.st_sflora.tables import headline_results, table1_top1_accuracy
from ltx_trainer.st_sflora.tokens import (
    batch_token_importance,
    cls_to_patch_attention,
    cumulative_semantic_retention,
    merge_discarded_tokens,
    payload_bits,
    topk_patch_indices,
)


def _toy_clients(n_patch: int = 196, batch: int = 8) -> list[ClientLinkState]:
    rng = np.random.default_rng(42)
    clients: list[ClientLinkState] = []
    for m in range(3):
        keys = rng.standard_normal((batch, n_patch, 64))
        q = rng.standard_normal((batch, 64))
        attn = cls_to_patch_attention(q, keys)
        imp = batch_token_importance(attn).tolist()
        clients.append(
            ClientLinkState(
                channel_gain=0.4 + 0.2 * m,
                distance_m=50.0 + 30.0 * m,
                velocity_mps=2.0 + m,
                forward_latency_s=0.15 + 0.02 * m,
                downlink_latency_s=0.01,
                importance_sorted=imp,
            )
        )
    return clients


def evaluation_smoke(cfg: StSfloraConfig | None = None) -> dict[str, Any]:
    c = cfg or StSfloraConfig()
    clients = _toy_clients(n_patch=c.num_patch_tokens, batch=8)
    selected = [
        client_selected(
            standing_time_s(500.0, st.distance_m, st.velocity_mps, 10.0),
            st.forward_latency_s + st.downlink_latency_s + 0.5,
        )
        for st in clients
    ]

    rng = np.random.default_rng(7)
    patches = rng.standard_normal((c.num_patch_tokens, c.embed_dim))
    attn_row = np.sort(rng.random(c.num_patch_tokens))[::-1]
    attn_row = attn_row / attn_row.sum()
    k_demo = 96
    idx = topk_patch_indices(attn_row, k_demo)
    merged = merge_discarded_tokens(patches, attn_row, idx)

    allocs, ste_joint, outer_iters = alternating_optimize(
        clients,
        bandwidth_total_hz=c.bandwidth_total_hz,
        p_max=c.p_max_w,
        e_max=0.05,
        noise_dbm_hz=c.noise_psd_dbm_hz,
        batch_size=c.batch_size,
        embed_dim=c.embed_dim,
        bits_per_element=c.bits_per_element,
        n_patch=c.num_patch_tokens,
        k_min=c.k_min,
        max_iters=c.max_outer_iters,
        tol_power=c.tol_power,
        tol_bandwidth=c.tol_bandwidth,
        tol_k=c.tol_k,
        tol_tau=c.tol_tau,
    )

    k_full = c.num_patch_tokens
    bits_full = payload_bits(
        k_full,
        batch_size=c.batch_size,
        embed_dim=c.embed_dim,
        bits_per_element=c.bits_per_element,
    )
    bits_sel = payload_bits(
        allocs[0].k_tokens if allocs else k_demo,
        batch_size=c.batch_size,
        embed_dim=c.embed_dim,
        bits_per_element=c.bits_per_element,
    )

    imp = clients[0].importance_sorted
    ste_static = semantic_transmission_efficiency(
        [cumulative_semantic_retention(imp, a.k_tokens) for a in allocs],
        [a.uplink_latency_s for a in allocs],
    )

    paper = table1_top1_accuracy()
    headlines = headline_results()

    return {
        "paper": c.paper_arxiv,
        "clients_selected": selected,
        "toy_merge_dim": int(merged.shape[0]),
        "payload_bits_full": int(bits_full),
        "payload_bits_selected_client0": int(bits_sel),
        "compression_ratio_payload": round(bits_full / max(bits_sel, 1), 3),
        "alternating_outer_iters": outer_iters,
        "ste_joint": round(ste_joint, 4),
        "ste_static_check": round(ste_static, 4),
        "client0_k_tokens": allocs[0].k_tokens if allocs else None,
        "paper_vit_b16_st_imagenet100_non_iid": paper["vit_b16_non_iid"]["ImageNet100"][
            "ST-SFLora"
        ],
        "paper_lowest_gpu_mem_gb": headlines["lowest_client_gpu_mem_gb"],
        "cut_layers_client": c.client_cut_layers,
        "lora_rank_default": 16,
    }
