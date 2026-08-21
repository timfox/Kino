"""Channel grouping for dual-branch fusion benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelGrouping:
    branch_a: tuple[int, ...]
    branch_b: tuple[int, ...]
    name: str = "custom"

    def validate(self, n_channels: int = 12) -> None:
        all_ch = set(self.branch_a) | set(self.branch_b)
        if len(all_ch) != n_channels:
            raise ValueError("grouping must partition all channels")
        if len(self.branch_a) != len(self.branch_b):
            raise ValueError("branches must be equal size")


# Table 4 rank-1 grouping
BEST_GROUPING = ChannelGrouping(
    branch_a=(1, 7, 8, 9, 10, 11),
    branch_b=(0, 2, 3, 4, 5, 6),
    name="rank1_cross",
)

DEFAULT_CONTIGUOUS = ChannelGrouping(
    branch_a=tuple(range(6)),
    branch_b=tuple(range(6, 12)),
    name="default_contiguous",
)


def split_channels(channels: list[list[float]], grouping: ChannelGrouping) -> tuple[list, list]:
    a = [channels[i] for i in grouping.branch_a]
    b = [channels[i] for i in grouping.branch_b]
    return a, b
