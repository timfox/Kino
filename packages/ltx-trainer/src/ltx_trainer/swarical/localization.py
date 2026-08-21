"""Online localization: HC, ISR, RSF (Sec. 4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.swarical.config import LocalizationMode
from ltx_trainer.swarical.planner import Point3


@dataclass
class PoseVector:
    """Relative pose ``ρ̂_u,v`` from child u to parent v (position only in stub)."""

    dx: float
    dy: float
    dz: float

    def magnitude(self) -> float:
        return (self.dx**2 + self.dy**2 + self.dz**2) ** 0.5


def correction_vector(ground: PoseVector, estimated: PoseVector) -> PoseVector:
    """``v_ij = ρ_ij − ρ̂_ij`` (Sec. 4)."""
    return PoseVector(
        ground.dx - estimated.dx,
        ground.dy - estimated.dy,
        ground.dz - estimated.dz,
    )


def average_correction(vectors: list[PoseVector]) -> PoseVector:
    """FLS moves along ``(1/N) Σ v_ij`` over reachable tree nodes."""
    if not vectors:
        return PoseVector(0.0, 0.0, 0.0)
    n = len(vectors)
    return PoseVector(
        sum(v.dx for v in vectors) / n,
        sum(v.dy for v in vectors) / n,
        sum(v.dz for v in vectors) / n,
    )


def intra_swarm_step(
    ground_poses: dict[int, PoseVector],
    estimated_poses: dict[int, PoseVector],
) -> PoseVector:
    """One intra-swarm localization iteration for FLS i."""
    corrections = [
        correction_vector(ground_poses[j], estimated_poses[j])
        for j in ground_poses
        if j in estimated_poses
    ]
    return average_correction(corrections)


def localization_mode_summary() -> dict[str, str]:
    return {
        LocalizationMode.HC.value: (
            "Highly Concurrent: primary localizes while anchor still moving; all swarms may localize together."
        ),
        LocalizationMode.ISR.value: (
            "Inter-Swarm Rounds: anchor stationary before child primary localizes; BFS on swarm-tree (recommended)."
        ),
        LocalizationMode.RSF.value: (
            "Rounds on FLS-trees: root notifies children in rounds; anchor stationary per edge."
        ),
    }


def is_converged(avg_vector: PoseVector, threshold: float) -> bool:
    return avg_vector.magnitude() < threshold


def apply_translation(points: list[Point3], dx: float, dy: float, dz: float) -> list[Point3]:
    return [Point3(p.x + dx, p.y + dy, p.z + dz) for p in points]


def center_align(source: list[Point3], target: list[Point3]) -> list[Point3]:
    """Translate source so centroids match target (Sec. 5.3)."""
    if not source or not target:
        return list(source)
    sx = sum(p.x for p in source) / len(source)
    sy = sum(p.y for p in source) / len(source)
    sz = sum(p.z for p in source) / len(source)
    tx = sum(p.x for p in target) / len(target)
    ty = sum(p.y for p in target) / len(target)
    tz = sum(p.z for p in target) / len(target)
    return apply_translation(source, tx - sx, ty - sy, tz - sz)
