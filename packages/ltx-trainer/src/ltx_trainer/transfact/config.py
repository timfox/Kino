"""TransFACT bovine embryo transferability constants (arXiv:2605.18923)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransfactConfig:
    paper_arxiv: str = "2605.18923"
    paper_title: str = (
        "From Division to Decision: Leveraging Temporal Cell-Stage Segmentation "
        "for Embryo Transferability Prediction"
    )
    model_name: str = "TransFACT"

    dataset_name: str = "INRAE-Gertrude-DT"
    num_videos: int = 1740
    split_train: int = 1220
    split_val: int = 175
    split_test: int = 345
    nt_class_fraction: float = 0.53

    num_frames: int = 300
    frame_interval_minutes: int = 15
    frame_size: tuple[int, int] = (256, 256)
    dpi_max: int = 4

    num_stage_classes: int = 11
    num_stage_tokens: int = 60
    num_update_blocks: int = 3

    classes_transferability: tuple[str, ...] = ("T", "NT")

    mhi_tau: int = 15
    mhi_theta: int = 20

    training_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"

    loss_weights: dict[str, float] | None = None

    baseline_competitor: str = "SFR"

    institutions: tuple[str, ...] = (
        "Inria Rennes / University of Rennes, IRISA",
        "Paris-Saclay, UVSQ, INRAE BREED / ENVA",
    )

    def __post_init__(self) -> None:
        if self.loss_weights is None:
            self.loss_weights = {
                "Ltrans": 1.0,
                "Lframe": 1.0,
                "Lstage": 1.0,
                "Lcross_att": 1.0,
                "Lsmooth": 5.0,
            }
