"""Colour themes (Tol palettes, Sec. 2.7)."""

from __future__ import annotations

# Paul Tol–style discrete colours (simplified hex)
THEME_RAINBOW = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
    "#EE7733",
    "#0077BB",
    "#33BBEE",
    "#EE3377",
    "#CC3311",
    "#009988",
)

THEME_IRIDESCENT = (
    "#4662AC",
    "#5A5FA8",
    "#6E5BA3",
    "#82579E",
    "#965399",
    "#AA4F94",
)

THEME_VIVID = (
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#FFFF33",
    "#A65628",
    "#F781BF",
)


def palette_for_theme(theme: str, n: int) -> list[str]:
    base = {
        "rainbow": THEME_RAINBOW,
        "iridescent": THEME_IRIDESCENT,
        "vivid": THEME_VIVID,
    }.get(theme, THEME_RAINBOW)
    return [base[i % len(base)] for i in range(n)]
