"""Getting a validation set into the shape the bundles can score.

Three input shapes are supported:

* **raw weatherAUS** - ``Date``, ``Location`` and the original observations.  This
  is the only shape that unlocks the geographic views, because it is the only one
  carrying ``Location``.  The app detects it (raw pressure is ~1000 hPa, scaled
  pressure is near 0), applies the recovered transform and builds all 44 features.
* **already-prepared** - a matrix that already holds model features.  The
  project's ``weatherAUS_X_test.csv`` is this shape, but it carries only 23 of the
  44 and no ``Location``.  The missing engineered columns **cannot** be rebuilt
  from it: the lag, window and neighbour terms live on the raw scale and this
  matrix is already scaled.
* **an upload** - either of the above, as CSV or Parquet.

Feed the widest history you have and slice afterwards with the date picker:
features are built over everything first, so lag and 3-day terms at the start of
the evaluation window still see the days before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import paths
import preprocessing

TARGET_CANDIDATES = [
    "RainTomorrow", "raintomorrow", "target", "y", "label", "y_true", "actual",
]
LOCATION_CANDIDATES = ["Location", "location", "station", "Station", "site", "Site"]

_TRUTHY = {"yes": 1, "no": 0, "true": 1, "false": 0, "y": 1, "n": 0, "1": 1, "0": 0}


@dataclass
class Coverage:
    present: list[str]
    missing: list[str]

    @property
    def is_scoreable(self) -> bool:
        return not self.missing


@dataclass
class ValidationData:
    frame: pd.DataFrame
    source: str
    target: pd.Series | None = None
    location_column: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def date_span(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if "Date" not in self.frame.columns:
            return None
        dates = pd.to_datetime(self.frame["Date"], errors="coerce").dropna()
        return (dates.min(), dates.max()) if not dates.empty else None

    def restrict_to_dates(self, start, end) -> ValidationData:
        """Slice the evaluation window; features were already built on the full history."""
        if "Date" not in self.frame.columns:
            return self
        dates = pd.to_datetime(self.frame["Date"], errors="coerce")
        keep = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        frame = self.frame.loc[keep].reset_index(drop=True)
        target = None if self.target is None else self.target.loc[keep].reset_index(drop=True)
        return ValidationData(
            frame=frame,
            source=self.source,
            target=target,
            location_column=self.location_column,
            notes=list(self.notes),
        )


def _first_present(frame: pd.DataFrame, candidates) -> str | None:
    lowered = {str(c).lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if str(candidate).lower() in lowered:
            return lowered[str(candidate).lower()]
    return None


def _coerce_target(values: pd.Series) -> pd.Series:
    if values.dtype == object:
        mapped = values.astype(str).str.strip().str.lower().map(_TRUTHY)
        return pd.to_numeric(mapped, errors="coerce")
    if values.dtype == bool:
        return values.astype(int)
    return pd.to_numeric(values, errors="coerce")


def read_table(handle, name: str = "") -> pd.DataFrame:
    """Read a CSV or Parquet upload, parsing Date when it is there."""
    if str(name).lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(handle)
    frame = pd.read_csv(handle)
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame


def bundled_raw() -> Path | None:
    """The project's raw history, if it is in place."""
    return paths.RAW_HISTORY if paths.RAW_HISTORY.exists() else None


def bundled_prepared() -> tuple[Path, Path] | None:
    """The project's prepared test split, as (features, labels)."""
    if paths.PREPARED_X.exists() and paths.PREPARED_Y.exists():
        return paths.PREPARED_X, paths.PREPARED_Y
    return None


def _pre_imputation() -> pd.DataFrame | None:
    """The pre-imputation frame the lag and neighbour terms should come from."""
    if not paths.PRE_IMPUTATION.exists():
        return None
    return pd.read_csv(paths.PRE_IMPUTATION, parse_dates=["Date"])


def build_validation_data(
    frame: pd.DataFrame,
    source: str,
    labels: pd.DataFrame | None = None,
    target_column: str | None = None,
) -> ValidationData:
    """Normalise any accepted input into a scoreable ValidationData."""
    frame = frame.reset_index(drop=True).copy()
    notes: list[str] = []

    target = None
    if labels is not None:
        column = target_column or _first_present(labels, TARGET_CANDIDATES) or labels.columns[0]
        target = _coerce_target(labels[column]).reset_index(drop=True)
        notes.append(f"Target taken from the labels file, column `{column}`.")
    else:
        column = target_column or _first_present(frame, TARGET_CANDIDATES)
        if column is not None:
            target = _coerce_target(frame[column])
            notes.append(f"Target taken from column `{column}`.")

    location_column = _first_present(frame, LOCATION_CANDIDATES)

    rebuilt = False
    if preprocessing.looks_raw(frame) and {"Date", "Location"} <= set(frame.columns):
        pre = _pre_imputation()
        order = frame.index.to_numpy()
        frame = frame.assign(_row=order)
        frame = preprocessing.build_features(frame, pre)
        if target is not None:
            target = target.reindex(frame["_row"].to_numpy()).reset_index(drop=True)
        frame = frame.drop(columns=["_row"])
        rebuilt = True
        notes.append(
            "Detected raw weatherAUS columns: applied the recovered training "
            "transform and rebuilt all 44 features."
            + ("" if pre is not None
               else "  (Pre-imputation frame absent, so lags were built on the "
                    "imputed values - slightly lower fidelity.)")
        )

    if not rebuilt and not preprocessing.looks_raw(frame):
        notes.append(
            "Input is already on the model scale; engineered columns are used as "
            "supplied and cannot be rebuilt from scaled inputs."
        )

    if target is not None and len(target) != len(frame):
        raise ValueError(
            f"the labels have {len(target):,} rows but the features have {len(frame):,}"
        )

    return ValidationData(
        frame=frame,
        source=source,
        target=target,
        location_column=location_column,
        notes=notes,
    )


def assess_coverage(frame: pd.DataFrame, features) -> Coverage:
    columns = set(frame.columns)
    present = [f for f in features if f in columns]
    missing = [f for f in features if f not in columns]
    return Coverage(present=present, missing=missing)


def build_matrix(frame: pd.DataFrame, features) -> np.ndarray:
    """The model matrix in the bundle's exact column order.

    Columns the frame does not have become NaN, which XGBoost sends down each
    tree's default branch.  That is only meaningful under the degraded policy -
    the app warns loudly when it happens.
    """
    matrix = np.empty((len(frame), len(features)), dtype=float)
    for index, name in enumerate(features):
        if name in frame.columns:
            matrix[:, index] = pd.to_numeric(frame[name], errors="coerce").to_numpy()
        else:
            matrix[:, index] = np.nan
    return matrix
