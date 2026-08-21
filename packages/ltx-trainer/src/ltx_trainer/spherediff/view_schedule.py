"""89-view schedule from paper appendix E (Sec. E)."""

from __future__ import annotations

from ltx_trainer.spherediff.config import NUM_VIEW_DIRECTIONS


def paper_view_directions() -> list[tuple[float, float]]:
    """
    (azimuth°, elevation°) pairs matching SphereDiff inference (89 views).
    """
    views: list[tuple[float, float]] = []
    for phi in (-90.0, 90.0):
        for k in range(4):
            views.append((360.0 / 4.0 * k, phi))
    for phi in (-77.5, 77.5):
        for k in range(8):
            views.append((360.0 / 8.0 * k, phi))
    for phi in (-45.0, 45.0):
        for k in range(11):
            views.append((360.0 / 11.0 * k, phi))
    for phi in (-22.5, 22.5):
        for k in range(14):
            views.append((360.0 / 14.0 * k, phi))
    for k in range(15):
        views.append((360.0 / 15.0 * k, 0.0))
    assert len(views) == NUM_VIEW_DIRECTIONS, len(views)
    return views
