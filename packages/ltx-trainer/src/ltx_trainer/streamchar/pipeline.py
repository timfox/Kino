"""StreamChar framework card, paper tables, and smoke demos (arXiv:2605.25659)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.streamchar.config import StreamCharConfig
from ltx_trainer.streamchar.dit import StreamCharDiT
from ltx_trainer.streamchar.orchestrator import OrchestratorConditionHead
from ltx_trainer.streamchar.pap import ProgressAwarePointer


def framework_card(cfg: StreamCharConfig | None = None) -> dict[str, Any]:
    cfg = cfg or StreamCharConfig()
    return {
        "name": "StreamChar",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "architecture": "LLM Orchestrator + joint audio-video DiT (WAN 2.2-5B)",
        "orchestrator": cfg.orchestrator_llm,
        "streaming": f"{cfg.frames_per_chunk} frames/chunk @ {cfg.fps} fps, {cfg.student_steps}-step student",
        "student_steps": cfg.student_steps,
        "distillation": f"Stage I DMD ({cfg.distill_stage1_steps} steps) + Stage II rollout ({cfg.distill_stage2_steps})",
        "latency_h100_sec": cfg.chunk_latency_sec,
        "playback_budget_sec": cfg.playback_budget_sec,
        "mechanisms": ["Progress-Aware Pointer (PAP)", "Sink-chunk memory", "Motion-frame conditioning"],
    }


def table_emtd_short_clip() -> dict[str, dict[str, float | None]]:
    """Table 1 — EMTD 10s protocol."""
    rows = {
        "OVI": {"sync_c": 7.183, "sync_d": 8.692, "fid": 21.092, "fvd": 287.88, "h_anat": 0.929, "h_id": 0.892, "wer": 10.458},
        "LTX-2": {"sync_c": 7.892, "sync_d": 7.979, "fid": 23.019, "fvd": 275.68, "h_anat": 0.912, "h_id": 0.921, "wer": 4.549},
        "MagiHuman": {"sync_c": 8.754, "sync_d": 7.156, "fid": 18.122, "fvd": 235.97, "h_anat": 0.913, "h_id": 0.981, "wer": 7.717},
        "Ours (base model)": {"sync_c": 7.427, "sync_d": 8.309, "fid": 17.987, "fvd": 248.16, "h_anat": 0.939, "h_id": 0.941, "wer": 3.539},
        "Ours (after distill)": {"sync_c": 8.126, "sync_d": 8.497, "fid": 18.963, "fvd": 289.091, "h_anat": 0.941, "h_id": 0.924, "wer": 3.649},
        "SoulX-FlashTalk": {"sync_c": 9.067, "sync_d": 7.461, "fid": 14.982, "fvd": 278.45, "h_anat": 0.923, "h_id": 0.973, "wer": None},
        "LiveAvatar": {"sync_c": 7.204, "sync_d": 8.556, "fid": 20.392, "fvd": 394.27, "h_anat": 0.924, "h_id": 0.979, "wer": None},
        "Ours (stage-2 one-chunk)": {"sync_c": 5.596, "sync_d": 9.280, "fid": 19.453, "fvd": 285.45, "h_anat": 0.937, "h_id": 0.923, "wer": 35.436},
        "Ours (distill stage-2 only)": {"sync_c": 6.788, "sync_d": 9.455, "fid": 17.446, "fvd": 265.40, "h_anat": 0.948, "h_id": 0.942, "wer": 5.756},
    }
    return rows


def table_long_horizon() -> dict[str, dict[str, float]]:
    """Table 2 — 5-minute streaming stability."""
    return {
        "Ours w/o sink chunk": {"sync_c": 8.052, "sync_d": 8.180, "dynamic": 1.0, "drift": 0.0304},
        "Ours": {"sync_c": 8.185, "sync_d": 8.388, "dynamic": 1.0, "drift": 0.0067},
        "FlashTalk": {"sync_c": 9.593, "sync_d": 7.249, "dynamic": 0.75, "drift": 0.0055},
        "FlashHead": {"sync_c": 7.419, "sync_d": 8.685, "dynamic": 0.05, "drift": 0.0088},
        "LiveAvatar": {"sync_c": 7.983, "sync_d": 8.005, "dynamic": 1.0, "drift": 0.0130},
    }


def training_step_demo(cfg: StreamCharConfig | None = None) -> dict[str, float]:
    """Smoke: orchestrator c_a + DiT flow loss + PAP endpoint."""
    cfg = cfg or StreamCharConfig()
    torch.manual_seed(8)
    b, c, t, h, w = 2, 8, 4, 16, 16
    z_v = torch.randn(b, c, t, h, w)
    z_a = torch.randn(b, c, t, h, w)
    z_ref = torch.randn(b, c, 1, h, w).expand(-1, -1, t, -1, -1)
    z_mot = torch.randn(b, c, t, h, w)
    tau = torch.tensor([0.3, 0.7])

    orch = OrchestratorConditionHead(audio_dim=c)
    x_a_t = torch.randn(b, t, c)
    c_a = orch(x_a_t, tau)

    dit = StreamCharDiT(channels=c, cfg=cfg)
    loss = dit.training_loss(z_v, z_a, z_ref, z_mot, c_a, tau)

    pap = ProgressAwarePointer(dim=c)
    txt = torch.randn(b, 32, c)
    s_hat = pap(txt, c_a, transcript_len=32)

    return {
        "dit_loss": float(loss.detach()),
        "c_a_shape_t": float(c_a.shape[1]),
        "pap_endpoint_mean": float(s_hat.detach().mean()),
        "student_steps": float(cfg.student_steps),
        "within_playback_budget": float(cfg.chunk_latency_sec < cfg.playback_budget_sec),
    }


def evaluation_demo(cfg: StreamCharConfig | None = None) -> dict[str, Any]:
    """Smoke: paper table ordering and design claims."""
    cfg = cfg or StreamCharConfig()
    step = training_step_demo(cfg)
    t1 = table_emtd_short_clip()
    t2 = table_long_horizon()

    joint = [m for m in t1 if t1[m]["wer"] is not None]
    best_wer = min(t1[m]["wer"] for m in joint if "Ours" in m and "one-chunk" not in m and "stage-2 only" not in m)

    return {
        **step,
        "ours_best_wer": t1["Ours (base model)"]["wer"] == best_wer,
        "distill_wer_near_base": abs(t1["Ours (after distill)"]["wer"] - t1["Ours (base model)"]["wer"]) < 0.2,
        "ours_best_fid_joint": t1["Ours (base model)"]["fid"] <= min(
            t1[m]["fid"] for m in joint if "Ours" not in m
        ),
        "sink_reduces_drift": t2["Ours"]["drift"] < t2["Ours w/o sink chunk"]["drift"],
        "ours_max_dynamic": t2["Ours"]["dynamic"] == 1.0,
        "multi_chunk_beats_one_chunk_wer": t1["Ours (after distill)"]["wer"] < t1["Ours (stage-2 one-chunk)"]["wer"],
        "realtime_on_budget": cfg.chunk_latency_sec < cfg.playback_budget_sec,
    }
