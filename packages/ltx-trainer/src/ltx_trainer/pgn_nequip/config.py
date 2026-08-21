"""PGN + NequIP configuration (Ghomsheh et al., arXiv:2605.30822)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PGNDesign:
    """Solvent-free PGN design parameters (Supplementary Table 1)."""

    core_diameter_nm: float = 10.0
    n_particles: int = 100
    phi_c: float = 0.1
    mw_kda: float = 5.0
    rg_over_d: float = 0.27
    rho_p_g_cm3: float = 1.0
    sigma_g_chains_nm2: float = 1.8


@dataclass(frozen=True)
class NequIPHyperparams:
    """Optimal NequIP settings (Methods)."""

    lmax: int = 3
    n_layers: int = 3
    rcut_over_d: float = 6.0
    train_frames: int = 4000
    val_frames: int = 500
    test_frames: int = 500


@dataclass
class PGNNequIPConfig:
    paper_arxiv: str = "2605.30822"
    design: PGNDesign = field(default_factory=PGNDesign)
    nequip: NequIPHyperparams = field(default_factory=NequIPHyperparams)
    dft_cost_relative: float = 1.0
    gnn_cost_relative: float = 1e-4
    mc_acceptance_target: float = 0.30
    equilibrium_mae_kbt: float = 1.08
