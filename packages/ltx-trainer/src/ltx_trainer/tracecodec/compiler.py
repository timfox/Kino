"""Deterministic packet compiler C(â, cΔt) — § 3.3."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ltx_trainer.tracecodec.actions import PacketAction, Proto, TcpCtrl, TimedAction


@dataclass
class FlowTemplate:
    flow_token: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int


@dataclass
class DirectionState:
    next_seq: int = 0
    next_ack: int = 0
    established: bool = False


@dataclass
class FlowState:
    per_dir: dict[int, DirectionState] = field(default_factory=lambda: {0: DirectionState(), 1: DirectionState()})


@dataclass
class RenderedPacket:
    flow_token: int
    proto: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    direction: int
    tcp_flags: str
    tcp_seq: int
    tcp_ack: int
    payload_len: int
    delta_t_ms: float
    legal: bool = True


class FlowTable:
    """Maps flow_token + proto + gen_index → synthetic endpoint template."""

    def __init__(self, *, salt: str = "tracecodec", ipv4_prefix: str = "10.0") -> None:
        self.salt = salt
        self.ipv4_prefix = ipv4_prefix
        self._cache: dict[tuple[int, str, int], FlowTemplate] = {}

    def resolve(self, flow_token: int, proto: Proto, gen_index: int = 0) -> FlowTemplate:
        key = (flow_token, proto.value, gen_index)
        if key not in self._cache:
            h = hashlib.sha256(f"{self.salt}:{flow_token}:{proto.value}:{gen_index}".encode()).hexdigest()
            a, b, c, d = (int(h[i : i + 2], 16) for i in range(0, 8, 2))
            e, f = int(h[8:10], 16), int(h[10:12], 16)
            src_ip = f"{self.ipv4_prefix}.{a % 254 + 1}.{b % 254 + 1}"
            dst_ip = f"{self.ipv4_prefix}.{c % 254 + 1}.{d % 254 + 1}"
            src_port = 1024 + (e * 256 + f) % 60000
            dst_port = 1024 + int(h[12:16], 16) % 60000
            self._cache[key] = FlowTemplate(flow_token, src_ip, dst_ip, src_port, dst_port)
        return self._cache[key]


def _flags_for_ctrl(ctrl: TcpCtrl) -> str:
    mapping = {
        TcpCtrl.SYN: "S",
        TcpCtrl.SYN_ACK: "SA",
        TcpCtrl.ACK: "A",
        TcpCtrl.DATA: "PA",
        TcpCtrl.FIN: "FA",
        TcpCtrl.RST: "R",
        TcpCtrl.OTHER: "",
    }
    return mapping.get(ctrl, "")


def _update_tcp_state(state: DirectionState, peer: DirectionState, action: PacketAction) -> tuple[int, int]:
    seq = state.next_seq
    ack = peer.next_seq if peer.established or action.tcp_ctrl == TcpCtrl.SYN_ACK else 0
    plen = action.l4_payload_len

    if action.tcp_ctrl == TcpCtrl.SYN:
        state.next_seq += 1
        state.established = False
    elif action.tcp_ctrl == TcpCtrl.SYN_ACK:
        state.next_seq += 1
        state.established = True
        peer.established = True
    elif action.tcp_ctrl in (TcpCtrl.DATA, TcpCtrl.ACK, TcpCtrl.FIN):
        if plen > 0:
            state.next_seq += plen
        if action.tcp_ack_adv > 0:
            peer.next_seq = max(peer.next_seq, ack + action.tcp_ack_adv)
    return seq, ack


def compile_action(
    timed: TimedAction,
    *,
    flow_table: FlowTable,
    flow_states: dict[int, FlowState],
) -> RenderedPacket:
    """Lower one packet action under explicit transport state."""
    action = timed.action
    tpl = flow_table.resolve(action.flow_token, action.proto)
    st = flow_states.setdefault(action.flow_token, FlowState())
    d = action.direction
    peer = 1 - d
    seq, ack = _update_tcp_state(st.per_dir[d], st.per_dir[peer], action)

    if action.direction == 0:
        src_ip, dst_ip = tpl.src_ip, tpl.dst_ip
        src_port, dst_port = tpl.src_port, tpl.dst_port
    else:
        src_ip, dst_ip = tpl.dst_ip, tpl.src_ip
        src_port, dst_port = tpl.dst_port, tpl.src_port

    return RenderedPacket(
        flow_token=action.flow_token,
        proto=action.proto.value,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        direction=action.direction,
        tcp_flags=_flags_for_ctrl(action.tcp_ctrl),
        tcp_seq=seq,
        tcp_ack=ack,
        payload_len=action.l4_payload_len,
        delta_t_ms=timed.delta_t_ms,
        legal=True,
    )


def compile_trace(
    timed_actions: list[TimedAction],
    *,
    flow_table: FlowTable | None = None,
) -> list[RenderedPacket]:
    """C(â_{1:T}, cΔt_{1:T}) → rendered PCAP rows."""
    table = flow_table or FlowTable()
    states: dict[int, FlowState] = {}
    return [compile_action(t, flow_table=table, flow_states=states) for t in timed_actions]
