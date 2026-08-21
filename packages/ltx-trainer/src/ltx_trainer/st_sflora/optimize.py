"""Alternating power, bandwidth, and token optimization (Sec. VI)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ltx_trainer.st_sflora.radio import (
    min_power_for_latency,
    noise_psd_watts_per_hz,
    uplink_energy_j,
    uplink_latency_s,
    uplink_rate_bps,
)
from ltx_trainer.st_sflora.tokens import cumulative_semantic_retention, payload_bits

LN2 = math.log(2.0)


@dataclass
class ClientLinkState:
    channel_gain: float
    distance_m: float
    velocity_mps: float
    forward_latency_s: float
    downlink_latency_s: float
    importance_sorted: list[float]


@dataclass
class ClientAllocation:
    k_tokens: int
    bandwidth_hz: float
    power_w: float
    uplink_latency_s: float
    uplink_energy_j: float
    retention: float
    feasible: bool


def _phi_energy(power_w: float, phi: float, kappa: float) -> float:
    return math.log(1.0 + phi * power_w) - kappa * power_w


def optimal_power(
    data_bits: float,
    bandwidth_hz: float,
    channel_gain: float,
    noise_psd: float,
    *,
    p_max: float,
    e_max: float,
    p_min: float,
    tol: float = 1e-5,
) -> tuple[float, bool]:
    """SUBP1 compact solution + bisection for energy cap (Alg. 2)."""
    if bandwidth_hz <= 0 or data_bits <= 0:
        return 0.0, False
    phi = channel_gain / (noise_psd * bandwidth_hz)
    kappa = data_bits * LN2 / (e_max * bandwidth_hz)
    rate_at_pmax = uplink_rate_bps(bandwidth_hz, p_max, channel_gain, noise_psd)
    e_at_pmax = uplink_energy_j(p_max, uplink_latency_s(data_bits, rate_at_pmax))
    if e_at_pmax <= e_max and p_max >= p_min:
        return p_max, True
    if kappa >= phi:
        return 0.0, False
    lo, hi = 0.0, p_max
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if _phi_energy(mid, phi, kappa) >= 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break
    p_bar = lo
    p_up = min(p_max, p_bar)
    if p_min > p_up:
        return 0.0, False
    return p_up, True


def bandwidth_for_rate(
    rate_bps: float,
    power_w: float,
    channel_gain: float,
    noise_psd: float,
    *,
    w_max: float,
    tol: float = 1e-6,
) -> float:
    """Invert R(W) via bisection (inner loop of Alg. 3)."""
    if rate_bps <= 0:
        return 0.0

    def rate(w: float) -> float:
        return uplink_rate_bps(w, power_w, channel_gain, noise_psd)

    if rate(w_max) < rate_bps:
        return w_max
    lo, hi = 1.0, w_max
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if rate(mid) >= rate_bps:
            hi = mid
        else:
            lo = mid
    return hi


def total_bandwidth_for_tau(
    clients: list[ClientLinkState],
    k_tokens: list[int],
    powers: list[float],
    tau_s: float,
    *,
    batch_size: int,
    embed_dim: int,
    bits_per_element: int,
    noise_psd: float,
    e_max: float,
    w_tot: float,
) -> float:
    total = 0.0
    for st, k_m, p_m in zip(clients, k_tokens, powers, strict=True):
        s_m = payload_bits(
            k_m,
            batch_size=batch_size,
            embed_dim=embed_dim,
            bits_per_element=bits_per_element,
        )
        beta = batch_size * embed_dim * bits_per_element
        r_ul = uplink_rate_bps(1.0, p_m, st.channel_gain, noise_psd)  # placeholder
        _ = r_ul
        t_stand = st.forward_latency_s  # reuse slot for T0 proxy in toy
        r_min = max(
            s_m / tau_s if tau_s > 0 else float("inf"),
            p_m * s_m / e_max if e_max > 0 else float("inf"),
            s_m / max(1e-9, tau_s),
        )
        w_min = bandwidth_for_rate(
            r_min,
            p_m,
            st.channel_gain,
            noise_psd,
            w_max=w_tot,
        )
        total += w_min
        _ = beta
    return total


def k_max_feasible(
    *,
    n_patch: int,
    k_min: int,
    rate_bps: float,
    power_w: float,
    beta_bits_per_token: float,
    e_max: float,
    tau_s: float,
    t_standing_minus_t0: float,
) -> int:
    """Eq. 41 — largest feasible K."""
    if rate_bps <= 0:
        return -1
    caps = [
        n_patch,
        e_max * rate_bps / (power_w * beta_bits_per_token) - 2,
        t_standing_minus_t0 * rate_bps / beta_bits_per_token - 2,
        tau_s * rate_bps / beta_bits_per_token - 2,
    ]
    k_cap = int(math.floor(min(caps)))
    if k_cap < k_min:
        return -1
    return k_cap


def alternating_optimize(
    clients: list[ClientLinkState],
    *,
    bandwidth_total_hz: float,
    p_max: float,
    e_max: float,
    noise_dbm_hz: float,
    batch_size: int,
    embed_dim: int,
    bits_per_element: int,
    n_patch: int,
    k_min: int,
    max_iters: int = 8,
    tol_power: float = 1e-4,
    tol_bandwidth: float = 1e-4,
    tol_k: float = 1.0,
    tol_tau: float = 1e-6,
) -> tuple[list[ClientAllocation], float, int]:
    """Alg. 4 — alternate SUBP1, SUBP2 (tau bisection), SUBP3."""
    noise = noise_psd_watts_per_hz(noise_dbm_hz)
    m = len(clients)
    if m == 0:
        return [], 0.0, 0

    imp = [list(c.importance_sorted) for c in clients]
    k = [min(k_min, n_patch) for _ in clients]
    w = [bandwidth_total_hz / m] * m
    p = [p_max * 0.5] * m
    tau = 1.0

    for it in range(max_iters):
        p_prev = list(p)
        w_prev = list(w)
        k_prev = list(k)
        tau_prev = tau

        # SUBP1 — power per client
        for i, st in enumerate(clients):
            s_bits = payload_bits(
                k[i],
                batch_size=batch_size,
                embed_dim=embed_dim,
                bits_per_element=bits_per_element,
            )
            t_max = max(1e-3, 5.0 - st.forward_latency_s)
            p_min = min_power_for_latency(
                s_bits, w[i], st.channel_gain, noise, t_max
            )
            p_i, ok = optimal_power(
                s_bits,
                w[i],
                st.channel_gain,
                noise,
                p_max=p_max,
                e_max=e_max,
                p_min=min(p_min, p_max),
            )
            p[i] = p_i if ok else p_max * 0.1

        # SUBP2 — tau via bisection on total W_min
        tau_lo, tau_hi = 1e-4, 60.0
        for _ in range(48):
            tau_mid = (tau_lo + tau_hi) / 2.0
            need = 0.0
            for i, st in enumerate(clients):
                s_bits = payload_bits(
                    k[i],
                    batch_size=batch_size,
                    embed_dim=embed_dim,
                    bits_per_element=bits_per_element,
                )
                r_min = s_bits / tau_mid
                need += bandwidth_for_rate(
                    r_min,
                    p[i],
                    st.channel_gain,
                    noise,
                    w_max=bandwidth_total_hz,
                )
            if need > bandwidth_total_hz:
                tau_lo = tau_mid
            else:
                tau_hi = tau_mid
        tau = tau_hi
        w_need: list[float] = []
        for i, st in enumerate(clients):
            s_bits = payload_bits(
                k[i],
                batch_size=batch_size,
                embed_dim=embed_dim,
                bits_per_element=bits_per_element,
            )
            r_min = s_bits / tau
            w_need.append(
                bandwidth_for_rate(
                    r_min,
                    p[i],
                    st.channel_gain,
                    noise,
                    w_max=bandwidth_total_hz,
                )
            )
        scale = bandwidth_total_hz / max(sum(w_need), 1e-9)
        w = [wi * scale for wi in w_need]

        # SUBP3 — token budget
        beta = batch_size * embed_dim * bits_per_element
        for i, st in enumerate(clients):
            r_ul = uplink_rate_bps(w[i], p[i], st.channel_gain, noise)
            k_i = k_max_feasible(
                n_patch=n_patch,
                k_min=k_min,
                rate_bps=r_ul,
                power_w=p[i],
                beta_bits_per_token=beta,
                e_max=e_max,
                tau_s=tau,
                t_standing_minus_t0=max(0.5, 4.0 - st.forward_latency_s),
            )
            k[i] = k_i if k_i >= k_min else k_min

        if (
            max(abs(a - b) for a, b in zip(p, p_prev, strict=True)) < tol_power
            and max(abs(a - b) for a, b in zip(w, w_prev, strict=True)) < tol_bandwidth
            and max(abs(a - b) for a, b in zip(k, k_prev, strict=True)) < tol_k
            and abs(tau - tau_prev) < tol_tau
        ):
            break

    beta = batch_size * embed_dim * bits_per_element
    out: list[ClientAllocation] = []
    retentions: list[float] = []
    lats: list[float] = []
    for i, st in enumerate(clients):
        s_bits = payload_bits(
            k[i],
            batch_size=batch_size,
            embed_dim=embed_dim,
            bits_per_element=bits_per_element,
        )
        r_ul = uplink_rate_bps(w[i], p[i], st.channel_gain, noise)
        lat = uplink_latency_s(s_bits, r_ul)
        ener = uplink_energy_j(p[i], lat)
        ret = cumulative_semantic_retention(imp[i], k[i])
        feasible = k[i] >= k_min and math.isfinite(lat)
        out.append(
            ClientAllocation(
                k_tokens=k[i],
                bandwidth_hz=w[i],
                power_w=p[i],
                uplink_latency_s=lat,
                uplink_energy_j=ener,
                retention=ret,
                feasible=feasible,
            )
        )
        retentions.append(ret)
        lats.append(lat)

    ste = sum(retentions) / max(lats) if lats else 0.0
    return out, ste, it + 1
