"""Runtime configuration for Deopt-Reopt workflow demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.deopt_reopt.constants import MODEL_O120, MODEL_Q235, TRIALS_PER_MODE


@dataclass
class DeoptReoptConfig:
    """Knobs for workflow simulation and table reproduction."""

    model: str = MODEL_O120
    workflow: str = "single_shot"  # single_shot | iterative
    trials: int = TRIALS_PER_MODE
    temperature: float = 0.5
    max_completion_tokens: int = 16384
    kernels: tuple[str, ...] = field(
        default_factory=lambda: (
            "conv2d",
            "bfft",
            "softmax",
            "bgemm",
            "dfspmm",
            "fft",
            "btdma",
            "ddgemm",
            "stencil",
            "spmm",
            "gemm",
            "spmv",
        )
    )

    def __post_init__(self) -> None:
        if self.model not in (MODEL_O120, MODEL_Q235):
            raise ValueError(f"model must be {MODEL_O120} or {MODEL_Q235}")
        if self.workflow not in ("single_shot", "iterative"):
            raise ValueError("workflow must be single_shot or iterative")
        if self.trials < 1:
            raise ValueError("trials must be positive")
