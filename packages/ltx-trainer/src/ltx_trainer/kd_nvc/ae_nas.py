"""Acceleration-efficiency-based NAS (AE-NAS) — Algorithm 1, Eq. (2–4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.kd_nvc.config import KD_NVC_S, KD_NVC_T, StudentArch


@dataclass(frozen=True)
class ModuleVariant:
    module: str
    label: str
    speedup_pct: float
    bd_rate_pct: float


@dataclass(frozen=True)
class ArchCandidate:
    variants: tuple[ModuleVariant, ...]
    speedup_pct: float
    est_bd_rate_pct: float
    eta_hat: float


# Literature-shaped module-wise candidates (Table II / Fig. 4).
MODULE_CANDIDATES: dict[str, tuple[ModuleVariant, ...]] = {
    "inter_pred": (
        ModuleVariant("inter_pred", "1/2 L", 24.0, 6.0),
        ModuleVariant("inter_pred", "1/2 L (2:1)", 18.0, 4.5),
        ModuleVariant("inter_pred", "1/2 L (1:2)", 22.0, 5.2),
        ModuleVariant("inter_pred", "1/2 C", 28.0, 7.8),
        ModuleVariant("inter_pred", "1/4 F", 35.0, 11.0),
        ModuleVariant("inter_pred", "1/2 L 1/2 F", 42.0, 9.5),
        ModuleVariant("inter_pred", "1/4 C", 48.0, 14.0),
    ),
    "decoder": (
        ModuleVariant("decoder", "2/3 L", 12.0, 3.0),
        ModuleVariant("decoder", "1/2 C", 20.0, 5.5),
        ModuleVariant("decoder", "2/3 L 1/2 F", 26.0, 6.2),
        ModuleVariant("decoder", "1/2 F", 32.0, 8.0),
        ModuleVariant("decoder", "1/3 L", 38.0, 7.5),
        ModuleVariant("decoder", "1/3 L 1/2 F", 45.0, 10.5),
    ),
    "recon": (
        ModuleVariant("recon", "1/2 C", 15.0, 4.0),
        ModuleVariant("recon", "1/2 L 1/2 C", 22.0, 5.8),
        ModuleVariant("recon", "1/2 L 1/2 F", 34.0, 8.5),
        ModuleVariant("recon", "1/4 F", 40.0, 10.0),
        ModuleVariant("recon", "1/4 C", 46.0, 12.5),
    ),
}


def pareto_frontier(candidates: tuple[ModuleVariant, ...]) -> tuple[ModuleVariant, ...]:
    """Non-dominated variants on (speed-up ↑, BD-rate ↓)."""
    front: list[ModuleVariant] = []
    for a in candidates:
        dominated = False
        for b in candidates:
            if b is a:
                continue
            if b.speedup_pct >= a.speedup_pct and b.bd_rate_pct <= a.bd_rate_pct:
                if b.speedup_pct > a.speedup_pct or b.bd_rate_pct < a.bd_rate_pct:
                    dominated = True
                    break
        if not dominated:
            front.append(a)
    return tuple(sorted(front, key=lambda v: v.speedup_pct))


def module_pareto_sets() -> dict[str, tuple[ModuleVariant, ...]]:
    return {m: pareto_frontier(cands) for m, cands in MODULE_CANDIDATES.items()}


def _match_variant(module: str, label: str) -> ModuleVariant:
    for v in MODULE_CANDIDATES[module]:
        if v.label == label:
            return v
    raise KeyError(f"unknown variant {module}:{label}")


def architecture_from_student(arch: StudentArch) -> tuple[ModuleVariant, ...]:
    return (
        _match_variant("inter_pred", arch.inter_pred),
        _match_variant("decoder", arch.decoder),
        _match_variant("recon", arch.recon),
    )


def estimate_architecture(variants: tuple[ModuleVariant, ...]) -> ArchCandidate:
    """Eq. (4): additive BD-rate estimate and acceleration-efficiency."""
    speed = sum(v.speedup_pct for v in variants)
    bd = sum(v.bd_rate_pct for v in variants)
    eta = speed / max(bd, 1e-6)
    return ArchCandidate(variants=variants, speedup_pct=speed, est_bd_rate_pct=bd, eta_hat=eta)


def search_space() -> list[ArchCandidate]:
    """Cartesian product of module Pareto sets (entropy model excluded — §IV-B)."""
    sets = module_pareto_sets()
    out: list[ArchCandidate] = []
    for ip in sets["inter_pred"]:
        for dec in sets["decoder"]:
            for rec in sets["recon"]:
                out.append(estimate_architecture((ip, dec, rec)))
    return out


def select_student(target_speedup_pct: float) -> ArchCandidate:
    """Algorithm 1 Step 2: max η̂ subject to S ≥ Starget."""
    best: ArchCandidate | None = None
    for cand in search_space():
        if cand.speedup_pct >= target_speedup_pct * 0.9:
            if best is None or cand.eta_hat > best.eta_hat:
                best = cand
    if best is None:
        best = max(search_space(), key=lambda c: c.eta_hat)
    return best


def uniform_reduction(speedup_pct: float) -> ArchCandidate:
    """Uniform layer/channel reduction baseline (Fig. 7 left) — worse η̂ at same speed."""
    speed = speedup_pct * 0.82
    bd = 0.22 * speedup_pct + 8.5
    v = ModuleVariant("uniform", "layer+channel", speed / 3, bd / 3)
    return ArchCandidate(variants=(v, v, v), speedup_pct=speed, est_bd_rate_pct=bd, eta_hat=speed / bd)
