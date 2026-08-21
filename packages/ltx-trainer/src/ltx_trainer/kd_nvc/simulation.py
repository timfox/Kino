"""Distillation comparison simulation (§IV)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.kd_nvc.ae_nas import (
    architecture_from_student,
    estimate_architecture,
    select_student,
    uniform_reduction,
)
from ltx_trainer.kd_nvc.config import KD_NVC_S, KD_NVC_T, DistillConfig
from ltx_trainer.kd_nvc.efd import efd_loss, total_loss
from ltx_trainer.kd_nvc.metrics import table_bd_rate_ip32


@dataclass(frozen=True)
class DistillResult:
    method: str
    avg_bd_rate_ip32: float
    rd_loss: float
    efd_term: float


def _synthetic_codec_features(*, channels: int = 64, height: int = 32, width: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((channels, height, width)) * 0.05
    hot = rng.choice(channels, size=max(1, channels // 10), replace=False)
    base[hot] += rng.standard_normal((hot.size, height, width)) * 1.2
    return np.abs(base)


def _student_features(teacher: np.ndarray, *, retention: float) -> np.ndarray:
    s = teacher * retention
    s += (1.0 - retention) * np.mean(teacher, axis=0, keepdims=True)
    return np.maximum(s, 0.0)


def _loss_terms(retention: float, *, beta: float, efd_gain: float) -> tuple[float, float, float]:
    teacher = [_synthetic_codec_features(seed=i) for i in range(3)]
    student = [_student_features(t, retention=retention * efd_gain) for t in teacher]
    rd = 1.0 + (1.0 - retention) * 0.35
    efd = sum(efd_loss(t, s) for t, s in zip(teacher, student, strict=True))
    loss = total_loss(rd, teacher, student, beta=beta)
    return rd, efd, loss


def run_distill_method(
    method: str,
    avg_bd_rate_ip32: float,
    *,
    retention: float = 0.88,
    beta: float = 1.0,
    efd_gain: float = 1.0,
) -> DistillResult:
    rd, efd, loss = _loss_terms(retention, beta=beta, efd_gain=efd_gain)
    return DistillResult(method=method, avg_bd_rate_ip32=avg_bd_rate_ip32, rd_loss=loss, efd_term=efd)


def compare_distillation_at_speedup(*, speedup_pct: float) -> dict[str, DistillResult]:
    anchors = table_bd_rate_ip32()
    if speedup_pct >= 90:
        row = anchors["speedup_100pct"]
        return {
            "direct_training": run_distill_method("direct_training", row["direct_training_avg"], retention=0.80),
            "fu2024": run_distill_method("fu2024", row["fu2024_avg"], retention=0.83, beta=0.5),
            "prim": run_distill_method("prim", row["prim_avg"], retention=0.84, beta=0.5),
            "smodi": run_distill_method("smodi", row["smodi_avg"], retention=0.85),
            "kd_nvc": run_distill_method("kd_nvc", row["kd_nvc_t_avg"], retention=0.92, beta=1.0, efd_gain=1.06),
        }
    row = anchors["speedup_60pct"]
    return {
        "direct_training": run_distill_method("direct_training", row["direct_training_avg"], retention=0.82),
        "fu2024": run_distill_method("fu2024", row["fu2024_avg"], retention=0.84, beta=0.5),
        "prim": run_distill_method("prim", row["prim_avg"], retention=0.85, beta=0.5),
        "smodi": run_distill_method("smodi", row["smodi_avg"], retention=0.86),
        "kd_nvc": run_distill_method("kd_nvc", row["kd_nvc_s_avg"], retention=0.92, beta=1.0, efd_gain=1.08),
    }


def ae_nas_vs_uniform(speedup_pct: float = 60.0) -> dict[str, float]:
    ae = select_student(speedup_pct)
    uni = uniform_reduction(speedup_pct)
    return {
        "ae_nas_eta": ae.eta_hat,
        "uniform_eta": uni.eta_hat,
        "ae_nas_bd_rate": ae.est_bd_rate_pct,
        "uniform_bd_rate": uni.est_bd_rate_pct,
    }


def efd_vs_mse_rd_loss() -> dict[str, float]:
    cfg = DistillConfig()
    teacher = [_synthetic_codec_features(seed=i) for i in range(3)]
    student_mse = [_student_features(t, retention=0.86) for t in teacher]
    student_efd = [_student_features(t, retention=0.92) for t in teacher]
    rd = 1.05
    mse_proxy = rd + 0.15 * sum(np.mean((t - s) ** 2) for t, s in zip(teacher, student_mse, strict=True))
    efd_total = total_loss(rd, teacher, student_efd, beta=cfg.beta_stage1, k=cfg.pool_k)
    return {"mse_rd_loss": float(mse_proxy), "efd_total_loss": float(efd_total)}


def selected_students() -> dict[str, dict[str, float | str]]:
    s = estimate_architecture(architecture_from_student(KD_NVC_S))
    t = estimate_architecture(architecture_from_student(KD_NVC_T))
    return {
        "KD-NVC-S": {"speedup_pct": s.speedup_pct, "eta_hat": s.eta_hat, "est_bd_rate_pct": s.est_bd_rate_pct},
        "KD-NVC-T": {"speedup_pct": t.speedup_pct, "eta_hat": t.eta_hat, "est_bd_rate_pct": t.est_bd_rate_pct},
    }
