"""Rebuild the 44 features the EXP-007 bundles expect, from raw weatherAUS rows.

The bundles are bare boosters with no preprocessing pipeline attached, so this
module reproduces the transform the training notebooks applied.  The constants it
needs live in ``resources/preprocessing_stats.json`` (regenerate with
``tools/fit_preprocessing.py``).

Two rules decide everything here, and getting either backwards costs several
points of AUC silently:

* the base weather columns are **scaled** with a ``RobustScaler`` fitted on the
  pre-2016 rows;
* every **engineered** column - diffs, lags, 3-day windows, upwind-neighbour
  aggregates - is computed on **raw, unscaled** values, so the scaling happens
  last, after the engineered block is already built.

``note.md`` section 3.4 adds a third rule: the lag and neighbour terms are built
from the *pre-imputation* frame (``weatherAUS_trend_grouped.csv``), so no value
invented by MICE ever propagates into a lag.  Passing ``pre_imputation`` honours
that; without it the engineered block falls back to the frame itself, which costs
roughly 0.002 ROC AUC.

Validated against the bundles' own recorded metrics on the 2016-01-01+ window:
the two chronological bundles score ROC AUC 0.8937 / 0.8942 against a reported
0.8966 / 0.8964.
"""

from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd

import paths
from locations import upwind_neighbours

# ---------------------------------------------------------------- the contract

BASE_SCALED = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]
CYCLIC = [
    "Date_sin", "Date_cos", "WindGustDir_sin", "WindGustDir_cos",
    "WindDir9am_sin", "WindDir9am_cos", "WindDir3pm_sin", "WindDir3pm_cos",
]
LAG1_SOURCES = [
    "Humidity3pm", "Pressure3pm", "Rainfall", "Cloud3pm",
    "Sunshine", "Temp3pm", "WindGustSpeed",
]
NEIGHBOUR_SPECS = [
    ("nbr_Rainfall_t", "Rainfall", 0),
    ("nbr_Rainfall_t1", "Rainfall", 1),
    ("nbr_Pressure3pm_t", "Pressure3pm", 0),
    ("nbr_Pressure3pm_t1", "Pressure3pm", 1),
    ("nbr_Humidity3pm_t", "Humidity3pm", 0),
    ("nbr_RainToday_t", "RainToday", 0),
]
ENGINEERED = (
    ["Pressure_diff", "Temp_diff"]
    + [f"{c}_lag1" for c in LAG1_SOURCES]
    + ["Pressure_3d_chg", "Humidity_3d_mean", "Rainfall_3d_sum",
       "RainToday_lag1", "days_since_rain"]
    + [name for name, _, _ in NEIGHBOUR_SPECS]
    + ["nearest_dist"]
)
FEATURES = BASE_SCALED + CYCLIC + ENGINEERED  # the 44, in the bundles' order

DIRECTION_COLUMNS = ["WindGustDir", "WindDir9am", "WindDir3pm"]
COMPASS_DEGREES = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}
DAYS_IN_YEAR = 365.25
ROLLING_WINDOW = 3


@functools.lru_cache(maxsize=1)
def load_stats() -> dict:
    with open(paths.PREPROCESSING_STATS, encoding="utf-8") as handle:
        return json.load(handle)


def looks_raw(frame: pd.DataFrame) -> bool:
    """Raw pressure sits near 1000 hPa; the scaled matrix has it near zero."""
    for column in ("Pressure9am", "Pressure3pm"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                return bool(values.abs().median() > 100)
    return False


# ------------------------------------------------------------------- the build


def _add_cyclic(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill in the cyclic encodings that a raw file does not carry."""
    out = frame
    if "Date" in out.columns and not {"Date_sin", "Date_cos"} <= set(out.columns):
        day = pd.to_datetime(out["Date"]).dt.dayofyear
        angle = 2 * np.pi * day / DAYS_IN_YEAR
        out["Date_sin"] = np.sin(angle)
        out["Date_cos"] = np.cos(angle)
    for column in DIRECTION_COLUMNS:
        if f"{column}_sin" in out.columns or column not in out.columns:
            continue
        bearing = out[column].map(COMPASS_DEGREES)
        out[f"{column}_sin"] = np.sin(np.radians(bearing))
        out[f"{column}_cos"] = np.cos(np.radians(bearing))
    return out


def _rain_today(frame: pd.DataFrame) -> pd.Series:
    """RainToday is bool in the pipeline files and Yes/No in a raw export."""
    if "RainToday" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    values = frame["RainToday"]
    if values.dtype == object:
        return values.map({"Yes": 1.0, "No": 0.0, True: 1.0, False: 0.0}).astype(float)
    return values.astype(float)


def _days_since_rain(flags: pd.Series) -> np.ndarray:
    position = np.arange(len(flags))
    last_wet = np.where(flags.to_numpy() > 0, position, np.nan)
    return position - pd.Series(last_wet).ffill().to_numpy()


def _engineered_block(source: pd.DataFrame) -> pd.DataFrame:
    """Every engineered column, computed on raw values, keyed by (Date, Location)."""
    frame = source.sort_values(["Location", "Date"], kind="mergesort")
    grouped = frame.groupby("Location", sort=False)
    rain = _rain_today(frame)

    out = pd.DataFrame({
        "Date": frame["Date"].to_numpy(),
        "Location": frame["Location"].to_numpy(),
    })
    out["Pressure_diff"] = (frame["Pressure3pm"] - frame["Pressure9am"]).to_numpy()
    out["Temp_diff"] = (frame["Temp3pm"] - frame["Temp9am"]).to_numpy()

    for column in LAG1_SOURCES:
        out[f"{column}_lag1"] = grouped[column].shift(1).to_numpy()
    out["RainToday_lag1"] = rain.groupby(frame["Location"], sort=False).shift(1).to_numpy()

    out["Pressure_3d_chg"] = (
        frame["Pressure3pm"] - grouped["Pressure3pm"].shift(ROLLING_WINDOW)
    ).to_numpy()

    # the 3-day window covers the days before today; today's own value is already
    # a feature in its own right, so including it again would just blur the two
    def _window(column: str, how: str) -> np.ndarray:
        return grouped[column].transform(
            lambda s: getattr(s.shift(1).rolling(ROLLING_WINDOW, min_periods=1), how)()
        ).to_numpy()

    out["Humidity_3d_mean"] = _window("Humidity3pm", "mean")
    out["Rainfall_3d_sum"] = _window("Rainfall", "sum")
    out["days_since_rain"] = (
        rain.groupby(frame["Location"], sort=False).transform(_days_since_rain).to_numpy()
    )

    return _neighbour_block(frame, out, rain)


def _neighbour_block(frame: pd.DataFrame, out: pd.DataFrame, rain: pd.Series) -> pd.DataFrame:
    """Aggregate the k nearest stations upwind (strictly west) of each one."""
    graph = upwind_neighbours(load_stats()["neighbour_graph"]["k"])
    out["nearest_dist"] = out["Location"].map({k: v[1] for k, v in graph.items()})

    wide = {
        column: frame.pivot_table(index="Date", columns="Location", values=column,
                                  aggfunc="mean")
        for column in ("Rainfall", "Pressure3pm", "Humidity3pm")
    }
    wide["RainToday"] = frame.assign(_rain=rain).pivot_table(
        index="Date", columns="Location", values="_rain", aggfunc="mean"
    )

    for name, column, lag in NEIGHBOUR_SPECS:
        table = wide[column]
        columns = {}
        for station, (neighbours, _) in graph.items():
            present = [n for n in neighbours if n in table.columns]
            columns[station] = (
                table[present].mean(axis=1) if present
                else pd.Series(np.nan, index=table.index)
            )
        aggregate = pd.DataFrame(columns)
        if lag:
            aggregate = aggregate.shift(lag)
        tidy = (
            aggregate.stack(future_stack=True).rename(name).reset_index()
            .rename(columns={"level_1": "Location"})
        )
        out = out.merge(tidy, on=["Date", "Location"], how="left")
    return out


def _scale(frame: pd.DataFrame) -> pd.DataFrame:
    scaler = load_stats()["scaler"]
    centre = dict(zip(scaler["columns"], scaler["center"]))
    spread = dict(zip(scaler["columns"], scaler["scale"]))
    for column in scaler["columns"]:
        if column in frame.columns:
            frame[column] = (frame[column] - centre[column]) / spread[column]
    return frame


def build_features(
    frame: pd.DataFrame,
    pre_imputation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Raw weatherAUS rows -> a frame carrying all 44 model features.

    ``frame`` supplies the base columns; ``pre_imputation``, when given, supplies
    the values the engineered block is built from (see the module docstring).
    Both are keyed on ``(Date, Location)``.
    """
    if "Date" not in frame.columns or "Location" not in frame.columns:
        raise ValueError("rebuilding features needs both `Date` and `Location`")

    base = frame.copy()
    base["Date"] = pd.to_datetime(base["Date"])
    base = base.sort_values("Date", kind="mergesort").reset_index(drop=True)
    base = _add_cyclic(base)

    source = base if pre_imputation is None else pre_imputation.copy()
    if pre_imputation is not None:
        source["Date"] = pd.to_datetime(source["Date"])
        source = _add_cyclic(source)

    engineered = _engineered_block(source)
    base = _scale(base)

    return base.merge(engineered, on=["Date", "Location"], how="left",
                      suffixes=("", "_eng"))


def missing_for(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f not in frame.columns]
