"""Toy multi-flow packet-action trace generator."""

from __future__ import annotations

import random

from ltx_trainer.tracecodec.actions import FlowContext, PacketAction, Proto, TcpCtrl, TimedAction


def _handshake(flow: int, *, payload_len: int = 512) -> list[TimedAction]:
    return [
        TimedAction(PacketAction(flow, tcp_ctrl=TcpCtrl.SYN, direction=0), 1.0),
        TimedAction(PacketAction(flow, tcp_ctrl=TcpCtrl.SYN_ACK, direction=1), 2.0),
        TimedAction(PacketAction(flow, tcp_ctrl=TcpCtrl.ACK, direction=0), 0.5),
        TimedAction(
            PacketAction(flow, tcp_ctrl=TcpCtrl.DATA, direction=0, l4_payload_len=payload_len),
            3.0,
        ),
        TimedAction(PacketAction(flow, tcp_ctrl=TcpCtrl.ACK, direction=1, tcp_ack_adv=payload_len), 0.3),
        TimedAction(PacketAction(flow, tcp_ctrl=TcpCtrl.FIN, direction=0), 10.0),
        TimedAction(PacketAction(flow, tcp_ctrl=TcpCtrl.ACK, direction=1), 0.2),
    ]


def sample_multi_flow_trace(
    *,
    n_flows: int = 4,
    seed: int = 0,
) -> list[TimedAction]:
    """Interleaved flows — preserves multi-flow structure for compiler smoke."""
    rng = random.Random(seed)
    traces: list[TimedAction] = []
    for f in range(n_flows):
        traces.extend(_handshake(f, payload_len=rng.randint(128, 1024)))
    rng.shuffle(traces)
    # Re-assign monotonic-ish timing after shuffle
    t = 0.0
    out: list[TimedAction] = []
    for item in traces:
        t += max(0.01, item.delta_t_ms * rng.uniform(0.5, 1.5))
        ctx = FlowContext(gap_bucket=min(7, int(t) % 8), pkt_count_bucket=item.action.flow_token % 4)
        act = PacketAction(
            flow_token=item.action.flow_token,
            proto=Proto.TCP,
            direction=item.action.direction,
            tcp_ctrl=item.action.tcp_ctrl,
            l4_payload_len=item.action.l4_payload_len,
            tcp_ack_adv=item.action.tcp_ack_adv,
            context=ctx,
        )
        out.append(TimedAction(act, t))
    return out
