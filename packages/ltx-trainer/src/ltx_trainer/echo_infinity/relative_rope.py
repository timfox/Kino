"""Unified Relative RoPE Recipe (Echo-Infinity §3.4, Eq. 3)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.echo_infinity.config import EchoInfinityConfig


@dataclass(frozen=True)
class RoPELayout:
    """Temporal RoPE ids for sink, memory queries, local window, and current chunk."""

    sink: tuple[int, ...]
    memory: tuple[int, ...]
    local: tuple[int, ...]
    current: tuple[int, ...]

    @property
    def all_ids(self) -> tuple[int, ...]:
        return self.sink + self.memory + self.local + self.current

    @property
    def max_id(self) -> int:
        ids = self.all_ids
        return max(ids) if ids else 0

    @property
    def min_id(self) -> int:
        ids = self.all_ids
        return min(ids) if ids else 0


def _backward_block(end_exclusive: int, count: int) -> tuple[int, ...]:
    """Assign ids [end_exclusive - count, ..., end_exclusive - 1]."""
    if count <= 0:
        return ()
    start = end_exclusive - count
    return tuple(range(start, end_exclusive))


def compute_rope_layout(
    f_star: int,
    chunk_size: int,
    *,
    num_sink: int,
    num_memory: int,
    num_local: int,
    fmax: int,
    has_memory: bool,
) -> RoPELayout:
    """Eq. (3): tile active temporal ids over [0, fmax] with sink anchored at 0.

    ``f_star`` is the global frame index at the start of the current chunk.
    """
    if num_sink < 0 or num_local < 0 or chunk_size < 1 or fmax < 0:
        raise ValueError("invalid RoPE layout parameters")

    r_end = min(f_star + chunk_size - 1, fmax)
    sink = tuple(range(num_sink))

    mem_count = num_memory if has_memory else 0
    local_count = num_local
    cur_count = chunk_size

    # Non-sink ids are packed backward from r_end (growth phase) or rotated in mature phase.
    total_non_sink = mem_count + local_count + cur_count
    if total_non_sink == 0:
        return RoPELayout(sink=sink, memory=(), local=(), current=())

    # Growth: ids increase until r_end reaches fmax; mature: r_end stays at fmax and older ids rotate.
    if r_end < fmax:
        # Growth phase — newest chunk ends at r_end, older blocks sit immediately before it.
        cur = _backward_block(r_end + 1, cur_count)
        local = _backward_block(cur[0] if cur else r_end + 1, local_count)
        mem = _backward_block(local[0] if local else (cur[0] if cur else r_end + 1), mem_count)
    else:
        # Mature phase — cap newest at fmax; shift non-sink block backward by one slot per step is
        # modeled externally via ``advance_mature_rope``; here we pack at the ceiling.
        cur = _backward_block(fmax + 1, cur_count)
        local = _backward_block(cur[0] if cur else fmax + 1, local_count)
        mem = _backward_block(local[0] if local else (cur[0] if cur else fmax + 1), mem_count)

    # Clamp and ensure sink starts at 0 (non-sink must be >= num_sink in relative recipe).
    def _clamp(ids: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(max(num_sink, min(fmax, i)) for i in ids)

    return RoPELayout(
        sink=sink,
        memory=_clamp(mem),
        local=_clamp(local),
        current=_clamp(cur),
    )


def advance_mature_rope(layout: RoPELayout, *, num_sink: int, fmax: int) -> RoPELayout:
    """Rotate non-sink ids backward by one while keeping sink at [0, NS-1] (mature phase)."""
    non_sink = list(layout.memory + layout.local + layout.current)
    if not non_sink:
        return layout
    rotated = [num_sink] + [max(num_sink, min(fmax, x - 1)) for x in non_sink[:-1]]
    # Re-split by original segment lengths
    n_mem = len(layout.memory)
    n_loc = len(layout.local)
    n_cur = len(layout.current)
    mem = tuple(rotated[:n_mem])
    loc = tuple(rotated[n_mem : n_mem + n_loc])
    cur = tuple(rotated[n_mem + n_loc : n_mem + n_loc + n_cur])
    return RoPELayout(sink=layout.sink, memory=mem, local=loc, current=cur)


def layout_for_step(
    f_star: int,
    cfg: EchoInfinityConfig,
    *,
    has_memory: bool,
) -> RoPELayout:
    return compute_rope_layout(
        f_star,
        cfg.chunk_size_frames,
        num_sink=cfg.num_sink_frames,
        num_memory=cfg.num_memory_query_frames,
        num_local=cfg.local_window_frames,
        fmax=cfg.fmax,
        has_memory=has_memory,
    )


def verify_layout_in_range(layout: RoPELayout, fmax: int) -> bool:
    if not layout.all_ids:
        return True
    return layout.min_id >= 0 and layout.max_id <= fmax
