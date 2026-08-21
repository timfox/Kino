"""TRACECODEC: compiler-backed neural codec for PCAP traces (arXiv:2605.29941)."""

from ltx_trainer.tracecodec.actions import (
    COARSE_FIELDS,
    FlowContext,
    PacketAction,
    Proto,
    TcpCtrl,
    TimedAction,
    action_to_vector,
    vector_to_action,
)
from ltx_trainer.tracecodec.codec import CodecOutput, TraceCodecStub, coarse_field_accuracy
from ltx_trainer.tracecodec.compiler import FlowTable, FlowState, RenderedPacket, compile_action, compile_trace
from ltx_trainer.tracecodec.config import TraceCodecConfig
from ltx_trainer.tracecodec.losses import kl_divergence, trace_codec_loss, training_step
from ltx_trainer.tracecodec.metrics import (
    fig4_transition_distances,
    fig5_multiflow_diagnostics,
    table1_positioning,
    table3_decoded_pcap_fidelity,
    table4_latent_interface,
    table5_ablations,
)
from ltx_trainer.tracecodec.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    training_step_demo,
)
from ltx_trainer.tracecodec.synthetic import sample_multi_flow_trace

__all__ = [
    "COARSE_FIELDS",
    "CodecOutput",
    "FlowContext",
    "FlowState",
    "FlowTable",
    "PacketAction",
    "Proto",
    "RenderedPacket",
    "TcpCtrl",
    "TimedAction",
    "TraceCodecConfig",
    "TraceCodecStub",
    "action_to_vector",
    "benchmark_manifest",
    "coarse_field_accuracy",
    "compile_action",
    "compile_trace",
    "evaluation_demo",
    "fig4_transition_distances",
    "fig5_multiflow_diagnostics",
    "framework_card",
    "kl_divergence",
    "paper_limitations",
    "sample_multi_flow_trace",
    "table1_positioning",
    "table3_decoded_pcap_fidelity",
    "table4_latent_interface",
    "table5_ablations",
    "trace_codec_loss",
    "training_step",
    "training_step_demo",
    "vector_to_action",
]
