"""Weather-station coordinates, name matching, and the upwind neighbour graph.

The catalogue is ``resources/locations.csv`` (49 stations).  Station names in the
weatherAUS data are run together (``AliceSprings``, ``CoffsHarbour``), but uploads
routinely spell them with spaces, underscores or different case, so matching is
done on a normalised key rather than the raw string.
"""

from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd

import paths

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km.  Broadcasts, so it builds a matrix at once."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@functools.lru_cache(maxsize=1)
def load_stations() -> pd.DataFrame:
    """The station catalogue, with a normalised join key."""
    frame = pd.read_csv(paths.LOCATIONS_CSV)
    frame["key"] = frame["Location"].map(_normalise)
    return frame


def attach_coordinates(frame: pd.DataFrame, location_column: str):
    """Add ``station``/``lat``/``lon`` to ``frame``; report names we could not place.

    Returns ``(joined, unmatched)``.  ``joined`` keeps every original row and
    column - unmatched rows simply carry NaN coordinates - so callers can align it
    positionally with the data they scored.
    """
    stations = load_stations()
    lookup = stations.set_index("key")[["Location", "lat", "lon"]]

    out = frame.reset_index(drop=True).copy()
    keys = out[location_column].map(_normalise)
    matched = lookup.reindex(keys.to_numpy())

    out["station"] = np.where(matched["Location"].notna().to_numpy(),
                              matched["Location"].to_numpy(),
                              out[location_column].to_numpy())
    out["lat"] = matched["lat"].to_numpy()
    out["lon"] = matched["lon"].to_numpy()

    unmatched = sorted(
        {str(v) for v, ok in zip(out[location_column], matched["Location"].notna()) if not ok}
    )
    return out, unmatched


@functools.lru_cache(maxsize=8)
def upwind_neighbours(k: int = 3) -> dict[str, tuple[tuple[str, ...], float]]:
    """For each station, its ``k`` nearest neighbours strictly to the west.

    Australian weather systems travel west to east, so the stations upwind of a
    given one are the informative ones - this is the ``west_strict`` rule recorded
    in ``preprocessing_stats.json``.  The float is the distance to the *nearest*
    western station, which the models see as ``nearest_dist``.
    """
    stations = load_stations()
    lat = stations["lat"].to_numpy()
    lon = stations["lon"].to_numpy()
    names = stations["Location"].to_numpy()

    distance = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])

    graph: dict[str, tuple[tuple[str, ...], float]] = {}
    for i, name in enumerate(names):
        west = np.where(lon < lon[i])[0]
        if west.size == 0:
            graph[name] = ((), float("nan"))
            continue
        order = west[np.argsort(distance[i, west])][:k]
        graph[name] = (tuple(names[order]), float(distance[i, order[0]]))
    return graph
