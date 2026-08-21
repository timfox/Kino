"""NEUROK: Neural Object Kinematics (Geng et al., arXiv:2605.30347)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NeurokConfig:
    paper_arxiv: str = "arXiv:2605.30347"
    project_page: str = "https://chen-geng.com/neurok"
    num_latent_tokens: int = 64
    token_dim: int = 32
    num_surface_samples: int = 512
    deformation_dim: int = 8  # dual-quaternion parameterization (paper supp.)
    vae_kl_weight: float = 0.01
    lr_homography_group: float = 1e-3
    lr_path_offsets: float = 1e-1
    active_subspace_dim: int = 8
    simulation_dt: float = 0.02
    simulation_steps: int = 50
    progressive_interval_iters: int = 100
    train_objects: int = 10_000
    test_partnet_mobility: bool = True
    baselines: tuple[str, ...] = (
        "NeuralDeformationGraphs",
        "SINGAPO",
        "FreeArt3D",
        "CANOR",
        "KeyPointDeformer",
        "PhysDreamer",
        "OmniPhysGS",
        "Pixie",
        "AnimateAnyMesh",
    )
    reference_generators: tuple[str, ...] = field(
        default_factory=lambda: ("veo3.1", "ltx2.3", "wan2.2")
    )
