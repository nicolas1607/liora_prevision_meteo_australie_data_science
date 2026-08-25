"""Chart colours.

The eight categorical slots are the validated default data-viz palette, in the
order that clears every adjacent-pair gate in both modes (worst adjacent CVD
delta-E 9.1 light / 8.4 dark; worst adjacent normal-vision 19.6 / 19.3).  The app
caps model selection at eight for exactly this reason - a ninth series would have
to invent a hue, which is never allowed.

Three light-mode slots sit below 3:1 against the surface, so the palette carries a
"relief required" obligation: every chart that uses it is accompanied by a legend
and a table view of the same numbers, which the app already renders.

Sequential encoding (a station's magnitude on the map) uses one hue light->dark.
Signed metrics use the blue<->red diverging pair with a neutral grey midpoint,
never a rainbow and never a hue at the midpoint.
"""

from __future__ import annotations

CATEGORICAL_LIGHT = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
CATEGORICAL_DARK = [
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
]

# one hue, light -> dark; the lightest step means "near zero" and may recede
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

# two poles that read as opposite, neutral grey between them
DIVERGING = ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f0a3a3", "#d03b3b", "#8f2020"]

NEUTRAL_LIGHT = "#f0efec"
NEUTRAL_DARK = "#383835"

SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"

TEXT_PRIMARY_LIGHT = "#0b0b0b"
TEXT_SECONDARY_LIGHT = "#52514e"
TEXT_PRIMARY_DARK = "#ffffff"
TEXT_SECONDARY_DARK = "#c3c2b7"

GRID_LIGHT = "#e6e5e1"
GRID_DARK = "#2f2f2d"

MAX_SERIES = len(CATEGORICAL_LIGHT)


def categorical(dark: bool = False) -> list[str]:
    """The fixed hue order. Never cycled - the app caps selection at MAX_SERIES."""
    return CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT


def surface(dark: bool = False) -> str:
    return SURFACE_DARK if dark else SURFACE_LIGHT


def text_primary(dark: bool = False) -> str:
    return TEXT_PRIMARY_DARK if dark else TEXT_PRIMARY_LIGHT


def text_secondary(dark: bool = False) -> str:
    return TEXT_SECONDARY_DARK if dark else TEXT_SECONDARY_LIGHT


def grid(dark: bool = False) -> str:
    return GRID_DARK if dark else GRID_LIGHT


def neutral(dark: bool = False) -> str:
    return NEUTRAL_DARK if dark else NEUTRAL_LIGHT


def assign(names) -> dict[str, str]:
    """Map each series name to a fixed slot, in order.

    Colour follows the entity, not its rank: callers pass a stable ordering so
    dropping a model never repaints the survivors.
    """
    return {n: CATEGORICAL_LIGHT[i % MAX_SERIES] for i, n in enumerate(names)}
