"""LA-LQR configuration (arXiv:2606.04775)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LQRWeights:
    """Per-task LQR cost matrices (scalar multiples of identity in paper)."""

    q: float = 1.0
    r_text: float = 75_000.0
    q_terminal: float = 1.0
    lambda_setpoint: float = 1.0


@dataclass
class LalqrConfig:
    paper_arxiv: str = "arXiv:2606.04775"
    models: tuple[str, ...] = ("Wan2.1-T2V-14B LightX2V", "HunyuanVideo-1.5 LightX2V")
    latent_rank: int = 64
    contrastive_pairs: int = 20
    svd_oversampling: int = 10
    svd_seed: int = 0xC057
    text_dim: int = 5120
    demo_activation_dim: int = 512
    horizon_layers: int = 4
    diffusion_timesteps: int = 4
    intervention: str = "text_only"  # text_only | video_only | text_and_video
    benchmarks: tuple[str, ...] = ("T2VSafetyBench", "SafeSora")
    wan_categories: tuple[str, ...] = (
        "Copyright & Trademarks",
        "Pornography",
        "Gore",
        "Public Figure",
        "Sequential Action Risk",
    )
    hunyuan_categories: tuple[str, ...] = (
        "Violence",
        "Terrorism",
        "Racism",
        "Sexual",
        "Animal Abuse",
    )
    wan_lqr: dict[str, LQRWeights] = field(
        default_factory=lambda: {
            "Copyright & Trademarks": LQRWeights(q=10.0, r_text=50_000.0, lambda_setpoint=3.0),
            "Pornography": LQRWeights(q=5.0, r_text=75_000.0, lambda_setpoint=1.0),
            "Gore": LQRWeights(q=10.0, r_text=75_000.0, lambda_setpoint=1.0),
            "Public Figure": LQRWeights(q=10.0, r_text=50_000.0, lambda_setpoint=1.0),
            "Sequential Action Risk": LQRWeights(q=5.0, r_text=75_000.0, lambda_setpoint=1.5),
        }
    )
    hunyuan_lqr: dict[str, LQRWeights] = field(
        default_factory=lambda: {
            "Violence": LQRWeights(q=10.0, r_text=250.0, lambda_setpoint=5.0),
            "Terrorism": LQRWeights(q=10.0, r_text=250.0, lambda_setpoint=5.0),
            "Racism": LQRWeights(q=10.0, r_text=250.0, lambda_setpoint=5.0),
            "Sexual": LQRWeights(q=10.0, r_text=250.0, lambda_setpoint=5.0),
            "Animal Abuse": LQRWeights(q=10.0, r_text=250.0, lambda_setpoint=5.0),
        }
    )

    def lqr_for(self, *, model: str, category: str) -> LQRWeights:
        if "Hunyuan" in model or model.lower().startswith("hunyuan"):
            return self.hunyuan_lqr.get(category, LQRWeights(q=10.0, r_text=250.0, lambda_setpoint=5.0))
        return self.wan_lqr.get(category, LQRWeights(q=5.0, r_text=75_000.0, lambda_setpoint=1.0))
