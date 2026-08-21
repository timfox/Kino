"""Princeton Benchmark + kangaroo meshes used in Swarical experiments (companion Sec. 2.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MeshDataset:
    """Metadata for a shape in ``assets/dataset/mesh`` (upstream Swarical repo)."""

    dataset_id: str
    display_name: str
    mesh_filename: str
    source: str
    notes: str = ""
    fls_count_approx: int | None = None
    downsampled_variant: str | None = None


# Companion paper Sec. 2.3 — five Princeton shapes + kangaroo; chess has downsampled variant.
MESH_DATASETS: tuple[MeshDataset, ...] = (
    MeshDataset(
        "chess",
        "Chess piece",
        "chess.ply",
        "Princeton Shape Benchmark [22]",
        notes="Original + down-sampled variant in reproduction.ipynb",
        downsampled_variant="chess_downsampled.ply",
    ),
    MeshDataset("dragon", "Dragon", "dragon.ply", "Princeton Shape Benchmark [22]"),
    MeshDataset("palm", "Palm", "palm.ply", "Princeton Shape Benchmark [22]"),
    MeshDataset(
        "skateboard",
        "Skateboard",
        "skateboard.ply",
        "Princeton Shape Benchmark [22]",
        fls_count_approx=1372,
        notes="Thousand-core ISR/HC/RSF experiments (Figs. 13–15)",
    ),
    MeshDataset("racecar", "Racecar", "racecar.ply", "Princeton Shape Benchmark [22]"),
    MeshDataset(
        "kangaroo",
        "Kangaroo",
        "kangaroo.ply",
        "free3d.com",
        notes="Additional shape beyond Princeton set",
    ),
    MeshDataset(
        "rectangle_4x4",
        "2D rectangle grid (small-scale)",
        "synthetic_4x4",
        "Companion Sec. 2.1",
        fls_count_approx=16,
        notes="16 FLS processes, 4×4 grid; laptop decentralized localization demo",
    ),
)


def list_mesh_datasets() -> list[dict[str, Any]]:
    """Serialize mesh registry for agents and reproduction guides."""
    return [
        {
            "id": d.dataset_id,
            "name": d.display_name,
            "mesh": d.mesh_filename,
            "source": d.source,
            "notes": d.notes,
            "fls_count_approx": d.fls_count_approx,
            "downsampled_variant": d.downsampled_variant,
            "mesh_dir_upstream": "assets/dataset/mesh",
        }
        for d in MESH_DATASETS
    ]


def get_mesh_dataset(dataset_id: str) -> MeshDataset | None:
    for d in MESH_DATASETS:
        if d.dataset_id == dataset_id:
            return d
    return None
