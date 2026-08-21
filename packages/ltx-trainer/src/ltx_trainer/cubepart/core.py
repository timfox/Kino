import numpy as np

def part_schema_meshes(part_names: list[str], seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [f"{name}_{rng.integers(0, 9999)}" for name in part_names]
