"""Paper tables and structural diagnostics — § 4, Appendices."""

from __future__ import annotations


def table1_positioning() -> list[dict[str, str]]:
    return [
        {"work": "ET-BERT", "learned_object": "datagram tokens", "pcap_decode": "not required", "timing": "not required", "state_multiflow": "not a decode state", "contract": "none"},
        {"work": "NetDiffusion", "learned_object": "packet-like image", "pcap_decode": "yes, native", "timing": "not a codec variable", "state_multiflow": "protocol-constrained synthesis", "contract": "protocol-constrained synthesis"},
        {"work": "NetSSM", "learned_object": "raw packet sequence", "pcap_decode": "yes, native", "timing": "generated with sequence", "state_multiflow": "implicit in generator", "contract": "direct generation + repair"},
        {"work": "TraceCodec", "learned_object": "timed packet actions + latent", "pcap_decode": "yes", "timing": "yes, explicit", "state_multiflow": "flow slots + compiler", "contract": "deterministic compiler under explicit state"},
    ]


def table3_decoded_pcap_fidelity() -> list[dict[str, str | float]]:
    """Table 3 — CICIDS2017 Monday + MAWI rows."""
    rows = [
        {"dataset": "CICIDS2017 Monday", "method": "TVAE", "tcp_event_pp": 72.73, "count_pct": 37.83, "proto_pct": 5.16, "iat_pct": 14.83, "flow_cnt_pct": 25.19, "flow_dur_pct": 45.36},
        {"dataset": "CICIDS2017 Monday", "method": "TabSyn-VAE", "tcp_event_pp": 66.74, "count_pct": 2.97, "proto_pct": 0.62, "iat_pct": 1.08, "flow_cnt_pct": 46.32, "flow_dur_pct": 34.76},
        {"dataset": "CICIDS2017 Monday", "method": "TTVAE", "tcp_event_pp": 53.19, "count_pct": 37.65, "proto_pct": 5.34, "iat_pct": 14.40, "flow_cnt_pct": 104.98, "flow_dur_pct": 51.03},
        {"dataset": "CICIDS2017 Monday", "method": "TRACECODEC", "tcp_event_pp": 14.51, "count_pct": 0.00, "proto_pct": 0.00, "iat_pct": 0.84, "flow_cnt_pct": 0.03, "flow_dur_pct": 9.23},
        {"dataset": "CICIDS2017 Monday", "method": "Oracle", "tcp_event_pp": 2.85, "count_pct": 0.00, "proto_pct": 0.00, "iat_pct": 0.04, "flow_cnt_pct": 0.00, "flow_dur_pct": 0.01},
        {"dataset": "MAWI 202004071400", "method": "TVAE", "tcp_event_pp": 51.53, "count_pct": 12.05, "proto_pct": 6.38, "iat_pct": 40.05, "flow_cnt_pct": 90.51, "flow_dur_pct": 28.44},
        {"dataset": "MAWI 202004071400", "method": "TRACECODEC", "tcp_event_pp": 2.54, "count_pct": 0.00, "proto_pct": 0.00, "iat_pct": 9.57, "flow_cnt_pct": 0.06, "flow_dur_pct": 2.62},
        {"dataset": "MAWI 202004071400", "method": "Oracle", "tcp_event_pp": 1.79, "count_pct": 0.00, "proto_pct": 0.00, "iat_pct": 0.01, "flow_cnt_pct": 0.00, "flow_dur_pct": 0.00},
    ]
    return rows


def table4_latent_interface() -> list[dict[str, str | float]]:
    return [
        {"representation": "TRACECODEC", "usable_suffixes_pct": 100.00, "valid_pkt_coverage_pct": 100.00, "ca_iat_tv_pct": 13.00, "ca_pkt_size_tv_pct": 18.85, "ca_flow_pkts_tv_pct": 39.35, "ca_sess_dur_tv_pct": 40.44},
        {"representation": "TVAE", "usable_suffixes_pct": 0.00, "valid_pkt_coverage_pct": 63.18, "ca_iat_tv_pct": 41.07, "ca_pkt_size_tv_pct": 49.76, "ca_flow_pkts_tv_pct": 64.80, "ca_sess_dur_tv_pct": 57.75},
        {"representation": "TabSyn-VAE", "usable_suffixes_pct": 7.03, "valid_pkt_coverage_pct": 98.29, "ca_iat_tv_pct": 16.09, "ca_pkt_size_tv_pct": 25.59, "ca_flow_pkts_tv_pct": 42.99, "ca_sess_dur_tv_pct": 42.59},
    ]


def table5_ablations() -> list[dict[str, str | float]]:
    return [
        {"variant": "Full TRACECODEC", "action_pct": 99.73, "tcp_event_pp": 13.79, "iat_pct": 0.59, "flow_cnt_pct": 0.40, "flow_dur_pct": 8.41, "kl_rate_nats_pkt": 167.91},
        {"variant": "W/O KL", "action_pct": 99.58, "tcp_event_pp": 14.57, "iat_pct": 0.49, "flow_cnt_pct": 0.41, "flow_dur_pct": 8.64, "kl_rate_nats_pkt": 5222.77},
        {"variant": "W/O fine timing", "action_pct": 99.77, "tcp_event_pp": 12.89, "iat_pct": 13.37, "flow_cnt_pct": 0.40, "flow_dur_pct": 5.90, "kl_rate_nats_pkt": 152.41},
        {"variant": "W/O TCP state cues", "action_pct": 99.95, "tcp_event_pp": 138.85, "iat_pct": 0.51, "flow_cnt_pct": 0.40, "flow_dur_pct": 8.10, "kl_rate_nats_pkt": 94.22},
    ]


def fig4_transition_distances() -> list[dict[str, str | float]]:
    return [
        {"method": "Ground Truth", "row_tv_distance": 0.0},
        {"method": "Oracle", "row_tv_distance": 3.66e-5},
        {"method": "TraceCodec", "row_tv_distance": 2.76e-5},
        {"method": "TVAE", "row_tv_distance": 0.353},
        {"method": "TabSyn-VAE", "row_tv_distance": 0.229},
        {"method": "TTVAE", "row_tv_distance": 0.350},
    ]


def fig5_multiflow_diagnostics() -> dict[str, float | int]:
    return {
        "reference_flow_switch_rate": 0.396,
        "tracecodec_flow_switch_rate": 0.396,
        "tvae_flow_switch_rate": 0.770,
        "tabsyn_flow_switch_rate": 0.800,
        "ttvae_flow_switch_rate": 0.901,
        "reference_mean_run_length": 2.53,
        "tracecodec_mean_run_length": 2.53,
        "tvae_mean_run_length": 1.30,
    }
