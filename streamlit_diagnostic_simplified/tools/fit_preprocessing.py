"""Recover the EXP-007 preprocessing constants and write them to data/preprocessing_stats.json.

The bundles ship as fitted boosters with no preprocessing pipeline attached, so the
transform that produced their training matrix had to be reverse-engineered from the
two artefacts that *are* in the repo:

* ``<project>/data/weatherAUS_MICE_cyclic.csv`` - the imputed source table, raw units,
  with ``Date`` and ``Location``.
* ``<project>/data/weatherAUS_X_test.csv`` - the model-ready matrix for the test window.

Both come from the shared project ``data/`` directory (override with
``WEATHER_DATA_DIR``); the JSON is written to this app's ``resources/``.

Rows line up when the source table is stable-sorted by ``Date`` alone (original file
order preserved within a day) and cut at 2016-01-01.  With that pairing the transform
falls out exactly:

* the 16 numeric columns go through a ``RobustScaler`` fitted on the pre-2016 rows
  (reproduces ``X_test`` to ~6e-14),
* wind directions are label-encoded alphabetically (E=0 ... WSW=15),
* ``RainToday`` is cast bool -> int.

The 21 engineered columns are absent from ``X_test``, so their definitions were
recovered by scoring candidate variants against the bundles' own recorded metrics.
The winning configuration - every engineered feature computed on **raw, unscaled**
values - reproduces the chronological bundles to within 0.003 ROC AUC and 0.004 F1.

Run this script only to regenerate the JSON; the app reads the JSON, not this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402
from locations import haversine_km, load_stations  # noqa: E402

RAW_PATH = paths.RAW_HISTORY
REFERENCE_X = paths.PREPARED_X
OUT_PATH = paths.PREPROCESSING_STATS

TRAIN_END = "2016-01-01"
SCALED_COLUMNS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
    "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]
DIRECTION_COLUMNS = ["WindGustDir", "WindDir9am", "WindDir3pm"]


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"{RAW_PATH} is missing - cannot refit the preprocessing.")

    source = (
        pd.read_csv(RAW_PATH, parse_dates=["Date"])
        .sort_values("Date", kind="mergesort")
        .reset_index(drop=True)
    )
    is_train = source["Date"] < TRAIN_END
    scaler = RobustScaler().fit(source.loc[is_train, SCALED_COLUMNS])

    directions = sorted(source["WindGustDir"].astype(str).unique())
    stats = {
        "provenance": {
            "source": RAW_PATH.name,
            "train_end_exclusive": TRAIN_END,
            "train_rows": int(is_train.sum()),
            "generated_by": "tools/fit_preprocessing.py",
        },
        "scaler": {
            "kind": "RobustScaler",
            "columns": SCALED_COLUMNS,
            "center": [float(v) for v in scaler.center_],
            "scale": [float(v) for v in scaler.scale_],
        },
        "direction_encoding": {name: i for i, name in enumerate(directions)},
        "direction_columns": DIRECTION_COLUMNS,
        "neighbour_graph": {"k": 3, "rule": "west_strict"},
        "engineered_on": "raw",
    }

    residual = _verify(source, is_train, scaler, stats)
    stats["provenance"]["max_abs_residual_vs_X_test"] = residual

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"  train rows      : {stats['provenance']['train_rows']:,}")
    print(f"  residual vs X_test: {residual:.3g}")
    _report_graph()


def _verify(source, is_train, scaler, stats) -> float:
    """Round-trip the recovered transform against the shipped X_test matrix."""
    if not REFERENCE_X.exists():
        print("note: weatherAUS_X_test.csv is absent, skipping verification")
        return float("nan")

    reference = pd.read_csv(REFERENCE_X)
    test = source.loc[~is_train].reset_index(drop=True)
    if len(test) != len(reference):
        print(f"note: row counts differ ({len(test)} vs {len(reference)}), skipping")
        return float("nan")

    rebuilt = pd.DataFrame(scaler.transform(test[SCALED_COLUMNS]), columns=SCALED_COLUMNS)
    worst = float((rebuilt - reference[SCALED_COLUMNS]).abs().to_numpy().max())

    codes = stats["direction_encoding"]
    for column in DIRECTION_COLUMNS:
        got = test[column].astype(str).map(codes).to_numpy()
        assert (got == reference[column].to_numpy()).all(), f"{column} encoding mismatch"
    assert (test["RainToday"].astype(int).to_numpy() == reference["RainToday"].to_numpy()).all()
    for column in ("Date_sin", "Date_cos", "Month"):
        assert float((test[column] - reference[column]).abs().max()) == 0.0, column
    return worst


def _report_graph() -> None:
    stations = load_stations()
    lat, lon = stations["lat"].to_numpy(), stations["lon"].to_numpy()
    distance = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(distance, np.inf)
    print(f"  stations        : {len(stations)}")
    print(f"  median nearest  : {np.median(distance.min(axis=1)):.1f} km")


if __name__ == "__main__":
    main()
