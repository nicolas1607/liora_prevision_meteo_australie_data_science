"""Folium layers: where the model is strong and where it is weak.

Circle **colour** is the selected metric, circle **size** is how many validation
days that station contributed - support is encoded rather than hidden, so a
station with 40 days never looks as authoritative as one with 500.  Stations whose
error rate sits more than two standard errors from the overall rate get a dashed
ring.

Colour follows the metric's job: a magnitude gets the sequential blue ramp, a
signed difference gets the blue<->red diverging pair with neutral grey at zero.
Never a rainbow, and never a hue at the diverging midpoint.
"""

from __future__ import annotations

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster  # noqa: F401  (HeatMap used below)

import palette
from evaluation import FLAG_Z, STATION_METRICS, is_signed_metric

# label -> CARTO raster style, or None for a keyless built-in folium basemap
TILES = {
    "Muted light": "light_all",
    "Muted dark": "dark_all",
    "Voyager": "rastertiles/voyager",
    "Terrain": None,
}
CARTO_URL = "https://{{s}}.basemaps.cartocdn.com/{style}/{{z}}/{{x}}/{{y}}{{r}}.png?key={key}"
CARTO_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
)


def tile_options(label: str, carto_key: str | None = None) -> dict:
    """``folium.Map`` tile kwargs for a basemap label.

    CARTO basemaps are requested with the API key when one is configured.
    Without it CARTO serves an "API KEY REQUIRED" placeholder tile, so every
    label falls back to OpenStreetMap instead.
    """
    style = TILES[label]
    if style is None or not carto_key:
        return {"tiles": "OpenStreetMap"}
    return {
        "tiles": CARTO_URL.format(style=style, key=carto_key),
        "attr": CARTO_ATTRIBUTION,
        "subdomains": "abcd",
        "max_zoom": 20,
    }

AUSTRALIA_CENTRE = (-27.0, 133.0)
DEFAULT_ZOOM = 4
MIN_RADIUS = 5
MAX_RADIUS = 18
FLAG_DASH = "6,4"


def _colormap(values: pd.Series, metric: str):
    """Sequential for magnitude, diverging for a signed difference."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return cm.LinearColormap([palette.SEQUENTIAL[2], palette.SEQUENTIAL[-2]],
                                 vmin=0.0, vmax=1.0)

    if is_signed_metric(metric):
        limit = float(np.abs(clean).max()) or 1.0
        scale = cm.LinearColormap(palette.DIVERGING, vmin=-limit, vmax=limit)
    else:
        scale = cm.LinearColormap(palette.SEQUENTIAL,
                                  vmin=float(clean.min()), vmax=float(clean.max()))
    scale.caption = STATION_METRICS.get(metric, {}).get("label", metric)
    return scale


def _radius(support: float, low: float, high: float) -> float:
    if not np.isfinite(support) or high <= low:
        return (MIN_RADIUS + MAX_RADIUS) / 2
    # square root keeps area roughly proportional to support
    fraction = np.sqrt((support - low) / (high - low))
    return MIN_RADIUS + fraction * (MAX_RADIUS - MIN_RADIUS)


def _popup(row: pd.Series) -> str:
    fields = [
        ("Validation days", f"{row.get('support', float('nan')):,.0f}"),
        ("Observed rain rate", f"{row.get('positive_rate', float('nan')):.3f}"),
        ("Error rate", f"{row.get('error_rate', float('nan')):.3f}"),
        ("vs overall", f"{row.get('error_rate_delta', float('nan')):+.3f}"),
        ("Error z", f"{row.get('error_z', float('nan')):+.2f}"),
        ("Precision", f"{row.get('precision', float('nan')):.3f}"),
        ("Recall", f"{row.get('recall', float('nan')):.3f}"),
        ("F1", f"{row.get('f1', float('nan')):.3f}"),
    ]
    rows = "".join(
        f"<tr><td style='padding:1px 8px 1px 0;color:#52514e'>{name}</td>"
        f"<td style='text-align:right'><b>{value}</b></td></tr>"
        for name, value in fields
    )
    return (
        f"<div style='font-family:system-ui,sans-serif;font-size:12px'>"
        f"<div style='font-weight:600;margin-bottom:4px'>{row['station']}</div>"
        f"<table>{rows}</table></div>"
    )


def _plot(layer, frame: pd.DataFrame, metric: str, scale, show_labels: bool) -> None:
    usable = frame.dropna(subset=["lat", "lon"])
    if usable.empty:
        return
    support = pd.to_numeric(usable.get("support"), errors="coerce")
    low, high = float(support.min()), float(support.max())

    for _, row in usable.iterrows():
        value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
        colour = scale(value) if np.isfinite(value) else palette.neutral()
        flagged = abs(float(row.get("error_z", 0) or 0)) >= FLAG_Z

        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=_radius(row.get("support", np.nan), low, high),
            color="#0b0b0b" if not flagged else palette.DIVERGING[-1],
            weight=2 if flagged else 1,
            dash_array=FLAG_DASH if flagged else None,
            opacity=0.9 if flagged else 0.45,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85,
            popup=folium.Popup(_popup(row), max_width=280),
            tooltip=f"{row['station']}: {value:.3f}" if np.isfinite(value) else row["station"],
        ).add_to(layer)

        if show_labels:
            folium.Marker(
                location=(row["lat"], row["lon"]),
                icon=folium.DivIcon(html=(
                    "<div style='font-family:system-ui,sans-serif;font-size:10px;"
                    "color:#0b0b0b;text-shadow:0 0 3px #fcfcfb;white-space:nowrap;"
                    f"transform:translate(10px,-7px)'>{row['station']}</div>"
                )),
            ).add_to(layer)


def build_map(
    frame: pd.DataFrame,
    metric: str,
    tiles: dict | None = None,
    show_labels: bool = False,
    show_heat: bool = False,
) -> folium.Map:
    fmap = folium.Map(location=AUSTRALIA_CENTRE, zoom_start=DEFAULT_ZOOM,
                      control_scale=True, **(tiles or tile_options("Muted light")))
    scale = _colormap(frame.get(metric, pd.Series(dtype=float)), metric)

    if show_heat:
        points = frame.dropna(subset=["lat", "lon"])
        weights = pd.to_numeric(points.get(metric), errors="coerce").fillna(0.0)
        if not points.empty:
            spread = float(weights.max() - weights.min()) or 1.0
            normalised = (weights - weights.min()) / spread
            HeatMap(
                [[lat, lon, float(w)] for lat, lon, w
                 in zip(points["lat"], points["lon"], normalised)],
                radius=28, blur=22, min_opacity=0.25,
            ).add_to(folium.FeatureGroup(name="Heat", show=True).add_to(fmap))

    _plot(fmap, frame, metric, scale, show_labels)
    scale.add_to(fmap)
    return fmap


def build_comparison_map(
    frames: dict[str, pd.DataFrame],
    metric: str,
    tiles: dict | None = None,
) -> folium.Map:
    """One toggleable layer per model, on a shared colour scale."""
    fmap = folium.Map(location=AUSTRALIA_CENTRE, zoom_start=DEFAULT_ZOOM,
                      control_scale=True, **(tiles or tile_options("Muted light")))

    combined = pd.concat(
        [f.get(metric, pd.Series(dtype=float)) for f in frames.values()], ignore_index=True
    )
    scale = _colormap(combined, metric)

    for index, (name, frame) in enumerate(frames.items()):
        layer = folium.FeatureGroup(name=name, show=index == 0)
        _plot(layer, frame, metric, scale, show_labels=False)
        layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    scale.add_to(fmap)
    return fmap
