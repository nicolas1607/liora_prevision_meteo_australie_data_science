"""One station-day of raw observations -> processed features -> each model's prediction.

The user types the *original*, unscaled observations for a station and a date.
Nothing is reimplemented here: the row is appended to a window of the project's
raw history and pushed through ``preprocessing.build_features`` - the very path
the validation features take - so the lag, 3-day-window and upwind-neighbour
terms come from the days and stations around it, and the base columns are
RobustScaler-scaled exactly as in training.

Blanks stay missing (NaN).  Tree models (XGBoost, LightGBM, CatBoost, decision
tree, random forest) score through them; linear models cannot, and the result
table says which entered columns blocked them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_io
import preprocessing
from locations import upwind_neighbours

# a year of history gives days_since_rain a long enough lookback for almost every
# station, while keeping the rebuild to a few thousand rows
HISTORY_WINDOW_DAYS = 365
RAIN_TODAY_MM = 1.0  # the dataset's definition: RainToday = Rainfall > 1 mm

COMPASS = list(preprocessing.COMPASS_DEGREES)

# raw input -> (label, unit, min, max, step), grouped as the form lays them out
RAW_FIELDS = {
    "Temperature": {
        "MinTemp": ("Min temp", "°C", -10.0, 50.0, 0.1),
        "MaxTemp": ("Max temp", "°C", -10.0, 50.0, 0.1),
        "Temp9am": ("Temp 9am", "°C", -10.0, 50.0, 0.1),
        "Temp3pm": ("Temp 3pm", "°C", -10.0, 50.0, 0.1),
    },
    "Rain & sun": {
        "Rainfall": ("Rainfall", "mm", 0.0, 400.0, 0.1),
        "Evaporation": ("Evaporation", "mm", 0.0, 150.0, 0.1),
        "Sunshine": ("Sunshine", "h", 0.0, 15.0, 0.1),
    },
    "Wind": {
        "WindGustSpeed": ("Gust speed", "km/h", 0.0, 150.0, 1.0),
        "WindSpeed9am": ("Speed 9am", "km/h", 0.0, 130.0, 1.0),
        "WindSpeed3pm": ("Speed 3pm", "km/h", 0.0, 130.0, 1.0),
    },
    "Humidity & pressure": {
        "Humidity9am": ("Humidity 9am", "%", 0.0, 100.0, 1.0),
        "Humidity3pm": ("Humidity 3pm", "%", 0.0, 100.0, 1.0),
        "Pressure9am": ("Pressure 9am", "hPa", 970.0, 1045.0, 0.1),
        "Pressure3pm": ("Pressure 3pm", "hPa", 970.0, 1045.0, 0.1),
    },
    "Cloud": {
        "Cloud9am": ("Cloud 9am", "oktas", 0.0, 9.0, 1.0),
        "Cloud3pm": ("Cloud 3pm", "oktas", 0.0, 9.0, 1.0),
    },
}
DIRECTION_FIELDS = {
    "WindGustDir": "Gust direction",
    "WindDir9am": "Direction 9am",
    "WindDir3pm": "Direction 3pm",
}
NUMERIC_INPUTS = [name for group in RAW_FIELDS.values() for name in group]


def rain_today(rainfall) -> float:
    """RainToday as the dataset defines it; unknown when Rainfall is blank."""
    if rainfall is None or pd.isna(rainfall):
        return np.nan
    return float(rainfall > RAIN_TODAY_MM)


def _user_row(location: str, date: pd.Timestamp, values: dict) -> pd.DataFrame:
    """The entered observations as one raw row, with its cyclic encodings."""
    row = {"Date": date, "Location": location}
    for name in NUMERIC_INPUTS:
        value = values.get(name)
        row[name] = np.nan if value is None else float(value)
    for name in DIRECTION_FIELDS:
        row[name] = values.get(name) or np.nan
    row["RainToday"] = rain_today(row["Rainfall"])
    frame = pd.DataFrame([row])
    # done on the row alone: the history already carries these columns, so
    # build_features would leave an appended row's encodings empty
    return preprocessing._add_cyclic(frame)


def _window(history: pd.DataFrame, date: pd.Timestamp, location: str) -> pd.DataFrame:
    dates = pd.to_datetime(history["Date"])
    keep = (dates >= date - pd.Timedelta(days=HISTORY_WINDOW_DAYS)) & (dates <= date)
    keep &= ~((dates == date) & (history["Location"] == location))
    window = history.loc[keep].copy()
    window["Date"] = pd.to_datetime(window["Date"])
    # booleans and the appended float must share one numeric dtype
    if "RainToday" in window.columns:
        window["RainToday"] = window["RainToday"].map(
            {True: 1.0, False: 0.0, "Yes": 1.0, "No": 0.0}
        ).astype(float)
    return window


def build_input_row(
    location: str,
    date,
    values: dict,
    history: pd.DataFrame,
    pre_imputation: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Process one station-day of raw values into the models' feature row.

    Returns the processed row (every column ``build_features`` produces, base
    columns on the training scale) and a context report on the history used.
    """
    date = pd.Timestamp(date).normalize()
    row = _user_row(location, date, values)

    base = pd.concat([_window(history, date, location), row], ignore_index=True)
    pre = None
    if pre_imputation is not None:
        pre = pd.concat([_window(pre_imputation, date, location), row], ignore_index=True)

    built = preprocessing.build_features(base, pre)
    mine = built[(built["Date"] == date) & (built["Location"] == location)]
    features = mine.drop(columns=[c for c in mine.columns if c.endswith("_eng")]).reset_index(drop=True)

    lookup = pre if pre is not None else base
    on_date = lookup.loc[lookup["Date"] == date, "Location"]
    previous = lookup.loc[lookup["Date"] == date - pd.Timedelta(days=1), "Location"]
    neighbours = upwind_neighbours(preprocessing.load_stats()["neighbour_graph"]["k"])
    upwind = neighbours.get(location, ((), np.nan))[0]
    history_end = pd.to_datetime(history["Date"]).max()

    context = {
        "history_end": history_end,
        "previous_day": bool((previous == location).any()),
        "history_days": int((lookup["Location"] == location).sum() - 1),
        "upwind": list(upwind),
        "upwind_with_data": [n for n in upwind if (on_date == n).any()],
        "rain_today": row["RainToday"].iloc[0],
    }
    return features, context


def predict(entries, models, names: dict[str, str], features: pd.DataFrame) -> pd.DataFrame:
    """Each loaded model's probability and decision for the processed row."""
    rows = []
    for entry, model in zip(entries, models):
        inputs = data_io.build_frame(features, model.input_features)
        record = {
            "Model": names[entry.key],
            "Folder": entry.folder,
            "Threshold": model.threshold,
        }
        try:
            proba = float(model.predict_proba(inputs)[0])
        except Exception as error:
            missing = [c for c in model.input_features if pd.isna(inputs[c].iloc[0])]
            record.update({
                "P(rain tomorrow)": np.nan,
                "Prediction": "-",
                "Status": (f"cannot score: missing {', '.join(missing)}" if missing
                           else f"cannot score: {type(error).__name__}"),
            })
        else:
            record.update({
                "P(rain tomorrow)": proba,
                "Prediction": "Rain" if proba >= model.threshold else "No rain",
                "Status": "ok",
            })
        rows.append(record)
    return pd.DataFrame(rows)


def processed_table(features: pd.DataFrame, models) -> pd.DataFrame:
    """Every model feature with its processed value and where it came from."""
    wanted: list[str] = []
    for model in models:
        wanted += [f for f in model.input_features if f not in wanted]
    order = [f for f in preprocessing.FEATURES if f in wanted] + [
        f for f in wanted if f not in preprocessing.FEATURES
    ]

    def _source(name: str, value) -> str:
        if pd.isna(value):
            return "missing"
        if name in preprocessing.BASE_SCALED:
            return "entered (scaled)"
        if name.startswith("Date_"):
            return "derived from the date"
        if name in preprocessing.CYCLIC:
            return "derived from the direction"
        if name in ("Pressure_diff", "Temp_diff"):
            return "computed from the entered values"
        if name == "nearest_dist":
            return "station geography"
        return "computed from history"

    rows = []
    for name in order:
        value = features[name].iloc[0] if name in features.columns else np.nan
        rows.append({
            "Feature": name,
            "Value": value,
            "Source": _source(name, value),
            "Used by": sum(name in m.input_features for m in models),
        })
    return pd.DataFrame(rows)
