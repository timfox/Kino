"""Framework card, demos, benchmark manifest (arXiv:2605.29941)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.tracecodec.codec import TraceCodecStub, coarse_field_accuracy
from ltx_trainer.tracecodec.compiler import compile_trace
from ltx_trainer.tracecodec.config import TraceCodecConfig
from ltx_trainer.tracecodec.losses import training_step
from ltx_trainer.tracecodec.metrics import (
    fig4_transition_distances,
    fig5_multiflow_diagnostics,
    table1_positioning,
    table3_decoded_pcap_fidelity,
    table4_latent_interface,
    table5_ablations,
)
from ltx_trainer.tracecodec.synthetic import sample_multi_flow_trace


def framework_card(cfg: TraceCodecConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TraceCodecConfig()
    return {
        "name": "TRACECODEC",
        "paper": cfg.paper_arxiv,
        "idea": "Compiler-backed neural codec: learn timed packet actions + latents; deterministic compiler owns TCP state and PCAP rendering.",
        "layers": [
            "lift: PCAP → timed packet actions + context + Δt",
            "codec: q(z|x,c,Δt), p(x,Δt|z,c) with staged coarse→detail decode",
            "compiler: FlowTable + TCP automaton → state-consistent PCAP",
        ],
        "decode_interface": "packet_action_ir",
        "baselines": ["TVAE", "TabSyn-VAE", "GOGGLE", "TTVAE"],
        "datasets": list(cfg.datasets),
        "objective": "L_recon + λ_Δt L_Δt + β L_KL",
    }


def paper_limitations() -> list[str]:
    return [
        "Synthetic payload bytes only — no application-layer semantic replay.",
        "Privacy reduces direct identifiers but timing/interaction patterns may still leak.",
        "Primary foundation validation on CICIDS2017 Monday + MAWI 202004071400.",
    ]


def benchmark_manifest(cfg: TraceCodecConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TraceCodecConfig()
    return {
        "datasets": list(cfg.datasets),
        "metrics": [
            "tcp_event_dist_pp",
            "count_pct",
            "proto_pct",
            "iat_pct",
            "flow_cnt_pct",
            "flow_dur_pct",
            "tcp_transition_tv",
            "flow_switch_rate",
            "usable_suffixes_pct",
        ],
        "non_repair_policy": "no endpoint inference, no TCP-state repair, no TraceCodec compiler for raw baselines",
    }


def evaluation_demo(*, cfg: TraceCodecConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TraceCodecConfig()
    torch.manual_seed(7)
    trace = sample_multi_flow_trace(n_flows=3, seed=7)
    rendered = compile_trace(trace)
    model = TraceCodecStub(cfg)
    losses = [training_step(model, t, cfg=cfg) for t in trace[:8]]
    mean_acc = sum(coarse_field_accuracy(t.action, model(t).recon) for t in trace[:8]) / min(8, len(trace))

    cic = next(r for r in table3_decoded_pcap_fidelity() if r["method"] == "TRACECODEC" and "CICIDS" in str(r["dataset"]))
    return {
        "n_packets_synthetic": len(trace),
        "n_flows": len({p.flow_token for p in rendered}),
        "all_packets_legal": all(p.legal for p in rendered),
        "mean_action_accuracy_stub": round(mean_acc, 3),
        "mean_codec_loss": round(sum(x["loss"] for x in losses) / len(losses), 4),
        "paper_cicids_flow_cnt_err_pct": cic["flow_cnt_pct"],
        "transition_distances": fig4_transition_distances(),
        "multiflow": fig5_multiflow_diagnostics(),
        "conclusion": "Packet-action + compiler decode preserves flow/TCP structure; raw-field baselines fragment under non-repair policy.",
    }


def training_step_demo(*, cfg: TraceCodecConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TraceCodecConfig()
    model = TraceCodecStub(cfg)
    timed = sample_multi_flow_trace(n_flows=1, seed=1)[0]
    losses = training_step(model, timed, cfg=cfg)
    pkt = compile_trace([timed])[0]
    return {
        "losses": {k: round(v, 4) if isinstance(v, float) else v for k, v in losses.items()},
        "compiled_tcp_flags": pkt.tcp_flags,
        "flow_token": pkt.flow_token,
    }
