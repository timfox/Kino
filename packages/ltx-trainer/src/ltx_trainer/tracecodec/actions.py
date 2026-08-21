"""Packet-action IR — Table 2, § 3.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TcpCtrl(str, Enum):
    SYN = "syn"
    SYN_ACK = "syn_ack"
    ACK = "ack"
    DATA = "data"
    FIN = "fin"
    RST = "rst"
    OTHER = "other"


class Proto(str, Enum):
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"


@dataclass
class FlowContext:
    """Runtime context c_t — bounded flow-local history."""

    gap_bucket: int = 0
    pkt_count_bucket: int = 0
    last_payload_bucket: int = 0
    ack_streak: int = 0
    last_dir: int = 0  # 0=outbound, 1=inbound


@dataclass
class PacketAction:
    """Timed packet action a_t (behavioral IR, not raw header)."""

    flow_token: int
    flow_evt: int = 0
    proto: Proto = Proto.TCP
    ip_family: int = 4
    direction: int = 0
    tcp_ctrl: TcpCtrl = TcpCtrl.DATA
    l4_payload_len: int = 0
    tcp_win: int = 65535
    tcp_ack_adv: int = 0
    ttl_res: int = 64
    icmp_type: int = 0
    icmp_code: int = 0
    context: FlowContext = field(default_factory=FlowContext)


@dataclass
class TimedAction:
    action: PacketAction
    delta_t_ms: float


COARSE_FIELDS = (
    "flow_token",
    "flow_evt",
    "proto",
    "ip_family",
    "direction",
    "tcp_ctrl",
)


def action_to_vector(action: PacketAction) -> dict[str, Any]:
    """Coarse + detail dict for reconstruction loss."""
    return {
        "flow_token": action.flow_token,
        "flow_evt": action.flow_evt,
        "proto": action.proto.value,
        "ip_family": action.ip_family,
        "direction": action.direction,
        "tcp_ctrl": action.tcp_ctrl.value,
        "l4_payload_len": action.l4_payload_len,
        "tcp_win": action.tcp_win,
        "tcp_ack_adv": action.tcp_ack_adv,
        "ttl_res": action.ttl_res,
    }


def vector_to_action(vec: dict[str, Any], *, context: FlowContext | None = None) -> PacketAction:
    return PacketAction(
        flow_token=int(vec["flow_token"]),
        flow_evt=int(vec.get("flow_evt", 0)),
        proto=Proto(str(vec.get("proto", "tcp"))),
        ip_family=int(vec.get("ip_family", 4)),
        direction=int(vec.get("direction", 0)),
        tcp_ctrl=TcpCtrl(str(vec.get("tcp_ctrl", "data"))),
        l4_payload_len=int(vec.get("l4_payload_len", 0)),
        tcp_win=int(vec.get("tcp_win", 65535)),
        tcp_ack_adv=int(vec.get("tcp_ack_adv", 0)),
        ttl_res=int(vec.get("ttl_res", 64)),
        context=context or FlowContext(),
    )
