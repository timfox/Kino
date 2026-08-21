"""Caspar bundle-adjustment smoke pipeline and demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.caspar.config import CasparConfig
from ltx_trainer.caspar.dabseg import build_sin_cos_sum_graph
from ltx_trainer.caspar.factors import Camera, Observation, numerical_jacobian, pose_apply, snavely_reprojection_residual, stack_bal_residuals
from ltx_trainer.caspar.memory import AccessPattern, blocked_struct_of_arrays
from ltx_trainer.caspar.metrics import benchmarks_bundle, table_symbolic_optimizations
from ltx_trainer.caspar.optimize import common_partial_cse, count_add_instructions
from ltx_trainer.caspar.reorder import estimate_register_pressure, reorder_calls
from ltx_trainer.caspar.solver import lm_solve


def framework_card(cfg: CasparConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CasparConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "upstream": cfg.upstream,
        "symforce_integration": "Builds on SymForce symbolic Lie groups + codegen",
        "components": [
            "DABSEG — bipartite call/value expression graph",
            "Hardware-mapped ops (norm3, sincos, fma, rcp/mulf decomposition)",
            "Common partial subexpression elimination (§II-B)",
            "Context-aware reciprocal sharing (§II-C)",
            "Adaptive reordering (R0,A0,…,K) to cut register pressure (§III)",
            "Accumulators + contribute calls for sum/product FMA scheduling",
            "Memory accessors: sequential, indexed, shared read/add, blocked SoA (§IV)",
            "LM + block Jacobi PCGNR, no Schur (§V; Wu et al.)",
            "Auto-generated CUDA kernels + Python/C++ bindings from factors",
        ],
        "bal_factor": "Snavely reprojection — 43 CUDA kernels from one @caslib.add_factor",
        "solver_defaults": {
            "initial_trust_region": cfg.initial_trust_region,
            "pcg_tolerance": cfg.pcg_tolerance,
        },
    }


def paper_limitations() -> list[str]:
    return [
        "This stub runs NumPy PCGNR — not generated CUDA or SymForce runtime.",
        "No nvcc/SASS inspection; hardware-mapped ops are labels only.",
        "BAL speedup tables are paper anchors, not reproduced GPU timings.",
        "Multi-GPU DABA and cuDSS linear solver paths are out of scope.",
    ]


def symbolic_demo() -> dict[str, Any]:
    """Partial CSE + DABSEG reorder smoke from paper examples."""
    sums = ["a+b+c", "a+c+d+e", "a+c+e"]
    optimized, temps = common_partial_cse(sums)
    cse_table = table_symbolic_optimizations()
    graph = build_sin_cos_sum_graph()
    order = reorder_calls(graph)
    reg_default = estimate_register_pressure(list(reversed(order)), graph)
    reg_opt = estimate_register_pressure(order, graph)
    return {
        "original_sums": sums,
        "caspar_optimized": optimized,
        "temps": temps,
        "original_fadd": count_add_instructions(sums),
        "caspar_fadd": count_add_instructions(list(optimized) + list(temps.values())),
        "paper_nvcc_fadd": cse_table["nvcc_cse_fadd"],
        "dabseg_call_order": order,
        "register_pressure_default": reg_default,
        "register_pressure_reordered": reg_opt,
    }


def _synthetic_bal(cfg: CasparConfig) -> tuple[list[Camera], np.ndarray, list[Observation]]:
    rng = np.random.default_rng(42)
    n_c = cfg.bal_smoke_cameras
    n_p = cfg.bal_smoke_points
    cameras: list[Camera] = []
    for i in range(n_c):
        T = np.eye(4)
        T[0, 3] = float(i) * 0.5
        T[2, 3] = 3.0
        cameras.append(Camera(T, focal=500.0, k1=0.01, k2=0.001))
    points = rng.normal(size=(n_p, 3)) * 0.5 + np.array([0.0, 0.0, 5.0])
    observations: list[Observation] = []
    for _ in range(cfg.bal_smoke_factors):
        ci = int(rng.integers(0, n_c))
        pi = int(rng.integers(0, n_p))
        p_cam = pose_apply(cameras[ci].cam_T_world, points[pi])
        d_safe = p_cam[2] + 1e-9 * np.sign(p_cam[2] if p_cam[2] != 0 else 1.0)
        p = -p_cam[:2] / d_safe
        p_norm2 = float(np.dot(p, p))
        r_dist = 1.0 + cameras[ci].k1 * p_norm2 + cameras[ci].k2 * p_norm2 * p_norm2
        pix_meas = cameras[ci].focal * r_dist * p + rng.normal(scale=0.3, size=2)
        observations.append(Observation(ci, pi, pix_meas))
    return cameras, points, observations


def bundle_adjustment_demo(cfg: CasparConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CasparConfig()
    cameras, points, observations = _synthetic_bal(cfg)
    J, r = numerical_jacobian(cameras, points, observations)

    def pack(cams: list[Camera], pts: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        for c in cams:
            parts.append(c.cam_T_world[:3, 3])
            parts.append(np.array([c.focal]))
        parts.append(pts.reshape(-1))
        return np.concatenate(parts)

    def unpack(x: np.ndarray) -> tuple[list[Camera], np.ndarray]:
        cams = [Camera(c.cam_T_world.copy(), c.focal, c.k1, c.k2) for c in cameras]
        off = 0
        for c in cams:
            c.cam_T_world[:3, 3] = x[off : off + 3]
            off += 3
            c.focal = float(x[off])
            off += 1
        pts = x[off:].reshape(points.shape)
        return cams, pts

    x0 = pack(cameras, points)

    def r_fn(x: np.ndarray) -> np.ndarray:
        c, p = unpack(x)
        return stack_bal_residuals(c, p, observations)

    def j_fn(x: np.ndarray) -> np.ndarray:
        c, p = unpack(x)
        Jx, _ = numerical_jacobian(c, p, observations)
        return Jx

    x_opt, stats = lm_solve(
        j_fn,
        r_fn,
        x0,
        initial_trust=cfg.initial_trust_region,
        pcg_tol=cfg.pcg_tolerance,
        max_iter=cfg.max_lm_iterations,
        block_size=6,
    )
    cams_opt, pts_opt = unpack(x_opt)
    final_r = stack_bal_residuals(cams_opt, pts_opt, observations)
    mse = float(np.mean(final_r**2))

    # Blocked SoA on camera records (focal, k1, k2 + t)
    cam_records = np.array([[c.focal, c.k1, c.k2, *c.cam_T_world[:3, 3]] for c in cameras])
    blocked, _ = blocked_struct_of_arrays(cam_records, chunk=cfg.vector_chunk_size)

    return {
        "n_cameras": len(cameras),
        "n_points": points.shape[0],
        "n_factors": len(observations),
        "jacobian_shape": list(J.shape),
        "residual_dim": int(r.size),
        "initial_mse": float(np.mean(r**2)),
        "final_mse": mse,
        "lm_iterations": stats.iterations,
        "pcg_iterations": stats.pcg_iters,
        "converged": stats.converged,
        "blocked_soa_shape": list(blocked.shape),
        "access_pattern": AccessPattern.SHARED_ADD.value,
        "generated_kernels_stub": 43,
    }


def evaluation_demo(cfg: CasparConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CasparConfig()
    return {
        "symbolic": symbolic_demo(),
        "bundle_adjustment": bundle_adjustment_demo(cfg),
        "benchmarks": benchmarks_bundle(),
    }


def evaluation_smoke(cfg: CasparConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CasparConfig()
    demo = evaluation_demo(cfg)
    sym = demo["symbolic"]
    ba = demo["bundle_adjustment"]
    ok = (
        sym["caspar_fadd"] <= sym["paper_nvcc_fadd"]
        and ba["final_mse"] <= ba["initial_mse"]
        and ba["jacobian_shape"][0] == ba["residual_dim"]
    )
    return {
        "package": "ltx_trainer.caspar",
        "ok": ok,
        "paper": cfg.paper_arxiv,
        "caspar_fadd": sym["caspar_fadd"],
        "ba_final_mse": ba["final_mse"],
        "framework": framework_card(cfg),
    }
