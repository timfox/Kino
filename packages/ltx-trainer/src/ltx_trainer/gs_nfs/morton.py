"""Morton codes and GPU-friendly octree occupancy (Sec. 2)."""

from __future__ import annotations

import numpy as np


def morton_encode(x: int, y: int, z: int, *, bits: int) -> int:
    """Bit-interleave voxel coordinates."""
    code = 0
    for b in range(bits):
        code |= ((x >> b) & 1) << (3 * b)
        code |= ((y >> b) & 1) << (3 * b + 1)
        code |= ((z >> b) & 1) << (3 * b + 2)
    return code


def morton_decode(code: int, *, bits: int) -> tuple[int, int, int]:
    x = y = z = 0
    for b in range(bits):
        x |= ((code >> (3 * b)) & 1) << b
        y |= ((code >> (3 * b + 1)) & 1) << b
        z |= ((code >> (3 * b + 2)) & 1) << b
    return x, y, z


def parent_code(child: int) -> int:
    return child >> 3


def child_base(parent: int) -> int:
    return parent << 3


def occupancy_byte(parent: int, children: np.ndarray) -> int:
    """Set bit i if child code parent<<3 + i exists in sorted children."""
    base = child_base(parent)
    occ = 0
    for c in children:
        if base <= c <= base + 7:
            occ |= 1 << (c - base)
    return occ


def build_level_lists(leaf_codes: np.ndarray, *, depth: int) -> list[np.ndarray]:
    """Bottom-up occupied Morton lists L0..LJ."""
    levels: list[np.ndarray] = [np.unique(np.sort(leaf_codes))]
    for _ in range(depth - 1, -1, -1):
        parents = np.unique(np.array([parent_code(c) for c in levels[-1]], dtype=np.int64))
        levels.append(parents)
    levels.reverse()
    return levels


def encode_occupancy_stream(levels: list[np.ndarray]) -> bytes:
    out: list[int] = []
    for d in range(len(levels) - 1):
        parents = levels[d]
        children = levels[d + 1]
        for p in parents:
            out.append(occupancy_byte(int(p), children))
    return bytes(out)


def decode_leaf_codes_bfs(occupancy: bytes, *, depth: int) -> np.ndarray:
    current = np.array([0], dtype=np.int64)
    ptr = 0
    for _ in range(depth):
        occ_slice = occupancy[ptr : ptr + len(current)]
        ptr += len(current)
        new_nodes: list[int] = []
        for parent, occ_byte in zip(current, occ_slice):
            base = child_base(int(parent))
            for i in range(8):
                if occ_byte & (1 << i):
                    new_nodes.append(base + i)
        current = np.array(new_nodes, dtype=np.int64)
    return current


def roundtrip_voxels(voxels: np.ndarray, *, bits: int) -> np.ndarray:
    """Encode/decode Morton occupancy and recover voxel coords."""
    codes = np.array([morton_encode(int(x), int(y), int(z), bits=bits) for x, y, z in voxels], dtype=np.int64)
    codes = np.unique(codes)
    levels = build_level_lists(codes, depth=bits)
    occ = encode_occupancy_stream(levels)
    decoded = decode_leaf_codes_bfs(occ, depth=bits)
    return np.array([morton_decode(int(c), bits=bits) for c in decoded], dtype=np.int64)
