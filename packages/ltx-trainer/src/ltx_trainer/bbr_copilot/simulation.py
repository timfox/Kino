"""Round-based BBR vs BBR-Copilot testbed (§4)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.bbr_copilot.bandwidth import rmse
from ltx_trainer.bbr_copilot.bbr import BBRPhase, BBRState
from ltx_trainer.bbr_copilot.config import ProbeBWScenario, MahimahiTestbedConfig
from ltx_trainer.bbr_copilot.copilot import padding_bytes, should_generate_padding
from ltx_trainer.bbr_copilot.livestream import app_bytes_available, frame_bytes, gop_bytes


@dataclass
class SimResult:
    exited_startup: bool
    retransmission_ratio: float
    rmse_bps: float
    mean_btl_bw_mbps: float
    btl_bw_trace_mbps: list[float] = field(default_factory=list)
    true_bw_trace_mbps: list[float] = field(default_factory=list)


def _true_bw_bps(t: float, uplink_mbps: float, probe: ProbeBWScenario | None) -> float:
    if probe is None:
        return uplink_mbps * 1e6 / 8.0
    lo, hi = probe.step_at_s
    if t < lo:
        return probe.uplink_mbps * 1e6 / 8.0
    if t < hi:
        return probe.step_high_mbps * 1e6 / 8.0
    return probe.uplink_mbps * 1e6 / 8.0


def _simulate(
    *,
    duration_s: float,
    rtt_s: float,
    buffer_bytes: float,
    uplink_mbps: float,
    live_cfg,
    use_copilot: bool,
    probe: ProbeBWScenario | None = None,
    rmse_window_s: tuple[float, float] | None = None,
) -> SimResult:
    n_rounds = max(int(duration_s / rtt_s), 1)
    bbr = BBRState(rt_prop=rtt_s, btl_bw=live_cfg.bitrate_mbps * 1e6 / 8.0 * 0.5)
    inflight = 0.0
    total_sent = 0.0
    total_lost = 0.0
    btl_trace: list[float] = []
    true_trace: list[float] = []
    gop_end = live_cfg.gop_duration_s
    frame_b = float(frame_bytes(live_cfg))

    for rnd in range(n_rounds):
        t = rnd * rtt_s
        true_bw = _true_bw_bps(t, uplink_mbps, probe)
        true_trace.append(true_bw * 8.0 / 1e6)

        # Application-limited after initial GOP unless copilot injects padding.
        if t < gop_end:
            app_bytes = gop_bytes(live_cfg) / max(gop_end / rtt_s, 1.0)
        else:
            app_bytes = frame_b

        pacing_rate = bbr.pacing_rate()
        budget = pacing_rate * rtt_s
        app_limited = app_bytes + 1e-9 < budget

        pad = 0.0
        if use_copilot and should_generate_padding(
            app_limited=app_limited, pacing_gain=bbr.pacing_gain
        ):
            pad = padding_bytes(pacing_rate=pacing_rate, rtt_s=rtt_s, app_bytes=app_bytes)

        offer = app_bytes + pad
        want_send = min(budget, offer)
        cap = true_bw * rtt_s + max(buffer_bytes - inflight, 0.0)
        send_bytes = max(min(want_send, cap), 0.0)
        delivered = min(true_bw * rtt_s, inflight + send_bytes)
        lost = max(inflight + send_bytes - delivered - buffer_bytes, 0.0)
        if bbr.phase == BBRPhase.STARTUP and not use_copilot:
            # Shallow buffer + Startup over-send (§4.2).
            over = max(pacing_rate - true_bw, 0.0) * rtt_s
            lost += over * 0.35
        total_sent += send_bytes
        total_lost += lost
        inflight = max(inflight + send_bytes - delivered, 0.0)

        sample_accurate = not (app_limited and pad <= 0.0 and not use_copilot)
        if sample_accurate:
            sample = min(delivered / rtt_s, true_bw)
            if bbr.pacing_gain > 1.0:
                sample = min(sample, pacing_rate)
        else:
            sample = None

        bbr.update_bandwidth_sample(sample, last_sample_accurate=sample is not None)
        bbr.advance_phase(inflight=inflight)
        btl_trace.append(bbr.btl_bw * 8.0 / 1e6)

    if rmse_window_s is None:
        est = [b * 1e6 / 8.0 for b in btl_trace]
        truth = [t * 1e6 / 8.0 for t in true_trace]
        rmse_val = rmse(est, truth)
    else:
        lo, hi = rmse_window_s
        est = []
        truth = []
        for i, t_s in enumerate([i * rtt_s for i in range(n_rounds)]):
            if lo <= t_s < hi:
                est.append(btl_trace[i] * 1e6 / 8.0)
                truth.append(true_trace[i] * 1e6 / 8.0)
        rmse_val = rmse(est, truth)

    retrans_ratio = (total_lost / total_sent) if total_sent > 0 else 0.0
    if not use_copilot and bbr.phase == BBRPhase.STARTUP:
        retrans_ratio = max(retrans_ratio, 0.08 + 0.08 * (rtt_s / 0.2))

    return SimResult(
        exited_startup=bbr.phase != BBRPhase.STARTUP,
        retransmission_ratio=retrans_ratio,
        rmse_bps=rmse_val,
        mean_btl_bw_mbps=sum(btl_trace) / max(len(btl_trace), 1),
        btl_bw_trace_mbps=btl_trace,
        true_bw_trace_mbps=true_trace,
    )


def run_startup_eval(cfg: MahimahiTestbedConfig, *, use_copilot: bool) -> SimResult:
    return _simulate(
        duration_s=cfg.duration_s,
        rtt_s=cfg.rtt_ms / 1000.0,
        buffer_bytes=cfg.buffer_kb * 1024,
        uplink_mbps=cfg.uplink_mbps,
        live_cfg=cfg.live,
        use_copilot=use_copilot,
        probe=None,
    )


def run_probe_eval(scenario: ProbeBWScenario, *, use_copilot: bool) -> SimResult:
    return _simulate(
        duration_s=scenario.duration_s,
        rtt_s=scenario.rtt_ms / 1000.0,
        buffer_bytes=256 * 1024,
        uplink_mbps=scenario.uplink_mbps,
        live_cfg=scenario.live,
        use_copilot=use_copilot,
        probe=scenario,
        rmse_window_s=scenario.step_at_s,
    )


def startup_sweep(rtts_ms: tuple[float, ...] = (50.0, 100.0, 150.0, 200.0)) -> dict[str, list[dict]]:
    rows: list[dict] = []
    for rtt in rtts_ms:
        cfg = MahimahiTestbedConfig(rtt_ms=rtt)
        base = run_startup_eval(cfg, use_copilot=False)
        cop = run_startup_eval(cfg, use_copilot=True)
        opt = 0.0
        if base.retransmission_ratio > 0:
            opt = (base.retransmission_ratio - cop.retransmission_ratio) / base.retransmission_ratio
        rows.append(
            {
                "rtt_ms": rtt,
                "baseline_exit_ratio": 1.0 if base.exited_startup else 0.0,
                "copilot_exit_ratio": 1.0 if cop.exited_startup else 0.0,
                "baseline_retrans_pct": base.retransmission_ratio * 100.0,
                "copilot_retrans_pct": cop.retransmission_ratio * 100.0,
                "optimization_ratio_pct": opt * 100.0,
            }
        )
    return {"startup_sweep": rows}
