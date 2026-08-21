"""TRACECODEC configuration (arXiv:2605.29941)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraceCodecConfig:
    paper_arxiv: str = "arXiv:2605.29941"
    latent_dim: int = 32
    embed_dim: int = 64
    max_flow_slots: int = 256
    # Variational objective L = L_recon + λ_Δt L_Δt + β L_KL (§ 3.4)
    lambda_dt: float = 1.0
    beta_kl: float = 0.001
    # Datasets (Appendix A)
    datasets: tuple[str, ...] = ("cic_ids2017_monday", "mawi_202004071400")
    # Compiler
    tcp_idle_gap_s: float = 60.0
    ipv4_prefix: str = "10.0"
    # Training budgets (Appendix F) — reference only in stub
    raw_baseline_steps: dict[str, int] = field(
        default_factory=lambda: {"tvae": 100_000, "tabsyn_vae": 15_000, "goggle": 100_000, "ttvae": 100_000}
    )
