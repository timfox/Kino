"""Quality-aware runtime planner (Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.pagedweight.config import PagedWeightConfig
from ltx_trainer.pagedweight.pages import WeightPageTable


@dataclass
class PlanAction:
    key: tuple[int, int, str]
    from_bits: int
    to_bits: int
    damage: float
    released: int

    @property
    def damage_per_byte(self) -> float:
        return self.damage / max(1, self.released)


@dataclass
class PlanResult:
    actions: list[PlanAction] = field(default_factory=list)
    target_bytes: int = 0
    released_bytes: int = 0
    pressure_triggered: bool = False

    def as_dict(self) -> dict:
        return {
            "n_actions": len(self.actions),
            "target_bytes": self.target_bytes,
            "released_bytes": self.released_bytes,
            "pressure_triggered": self.pressure_triggered,
            "actions": [
                {
                    "key": list(a.key),
                    "from_bits": a.from_bits,
                    "to_bits": a.to_bits,
                    "damage": round(a.damage, 4),
                    "released": a.released,
                }
                for a in self.actions
            ],
        }


def routing_bucket(mass: float) -> int:
    """Bucket 0=hot … 2=cold."""
    if mass >= 0.4:
        return 0
    if mass >= 0.15:
        return 1
    return 2


BUCKET_MULTIPLIER = {0: 2.0, 1: 1.25, 2: 1.0}
BITWIDTH_FLOOR = {0: 5, 1: 4, 2: 3}


@dataclass
class QualityAwarePlanner:
    config: PagedWeightConfig = field(default_factory=PagedWeightConfig)

    def target_bytes_from_kv_pressure(self, free_blocks: int) -> int:
        """Eq. (7): D = max{0, Tblk+1 − Fblk} · BKV."""
        cfg = self.config
        return max(0, cfg.free_block_threshold + 1 - free_blocks) * cfg.kv_block_bytes

    def score_actions(
        self,
        table: WeightPageTable,
        *,
        prompt_norms: tuple[float, float, float] | None = None,
    ) -> list[PlanAction]:
        """Legal bitwidth reductions scored by prompt-adjusted damage / released bytes."""
        cfg = self.config
        actions: list[PlanAction] = []
        mean_n, rms_n, max_n = prompt_norms or (1.0, 1.0, 1.0)
        residual_feat = 0.3 * mean_n + 0.4 * rms_n + 0.3 * max_n

        for page in table.pages.values():
            b = page.committed_bits
            bucket = routing_bucket(page.routing_mass) if cfg.enable_routing else 2
            floor = BITWIDTH_FLOOR[bucket]
            mu = BUCKET_MULTIPLIER[bucket] if cfg.enable_routing else 1.0
            for b2 in page.supported:
                if b2 >= b or b2 < floor:
                    continue
                if not cfg.enable_global_sensitivity:
                    g = 1.0
                else:
                    g = page.global_damage(b, b2)
                if cfg.enable_prompt_residual:
                    eta = max(0.5, min(2.0, residual_feat * (1.0 + cfg.residual_strength * (b - b2) * 0.1)))
                else:
                    eta = 1.0
                damage = max(1e-6, mu * g * eta)
                released = page.released_bytes(b, b2)
                if released <= 0:
                    continue
                actions.append(
                    PlanAction(key=page.key, from_bits=b, to_bits=b2, damage=damage, released=released)
                )
        actions.sort(key=lambda a: a.damage_per_byte)
        return actions

    def greedy_select(self, actions: list[PlanAction], target: int) -> list[PlanAction]:
        chosen: list[PlanAction] = []
        used_keys: set[tuple[int, int, str]] = set()
        released = 0
        for a in actions:
            if a.key in used_keys:
                continue
            chosen.append(a)
            used_keys.add(a.key)
            released += a.released
            if released >= target:
                break
        return chosen

    def plan_step(
        self,
        table: WeightPageTable,
        *,
        free_blocks: int,
        prompt_norms: tuple[float, float, float] | None = None,
    ) -> PlanResult:
        target = self.target_bytes_from_kv_pressure(free_blocks)
        triggered = target > 0
        result = PlanResult(target_bytes=target, pressure_triggered=triggered)
        if not triggered or not self.config.enable_page_movement:
            return result
        scored = self.score_actions(table, prompt_norms=prompt_norms)
        selected = self.greedy_select(scored, target)
        for a in selected:
            table.set_desired(a.key, a.to_bits)
        result.actions = selected
        result.released_bytes = sum(a.released for a in selected)
        return result
