"""Single source of truth for where this app reads data, models and its own files.

The app used to carry private copies of the dataset and the model bundles. It now
reads the project's shared ``data/`` and ``models/`` directories, using the same two
environment overrides as the notebooks in ``notebooks/XGBoost_Sath/`` so a Colab or
relocated checkout can redirect both consistently:

    WEATHER_DATA_DIR    default <project>/data
    WEATHER_MODEL_DIR   default <project>/models

Files the app owns rather than shares - the station coordinates and the recovered
preprocessing constants - stay in ``streamlit_diagnostic/resources/`` and are not
affected by those variables.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent


def _env_dir(variable: str, default: Path) -> Path:
    override = os.environ.get(variable)
    return Path(override).expanduser().resolve() if override else default


DATA_DIR = _env_dir("WEATHER_DATA_DIR", PROJECT_ROOT / "data")
MODEL_DIR = _env_dir("WEATHER_MODEL_DIR", PROJECT_ROOT / "models")
RESOURCE_DIR = APP_ROOT / "resources"

# --- shared, from the project pipeline (gitignored; see README) ----------------
RAW_HISTORY = DATA_DIR / "weatherAUS_MICE_cyclic.csv"
PREPARED_X = DATA_DIR / "weatherAUS_X_test.csv"
PREPARED_Y = DATA_DIR / "weatherAUS_y_test.csv"
# pre-imputation frame: the lag and neighbour features are built from this so no
# MICE-invented value propagates into a lag (note.md 3.4).  Optional - without it
# the engineered block falls back to RAW_HISTORY, at a small cost in fidelity.
PRE_IMPUTATION = DATA_DIR / "weatherAUS_trend_grouped.csv"

# --- app-owned reference files -------------------------------------------------
LOCATIONS_CSV = RESOURCE_DIR / "locations.csv"
PREPROCESSING_STATS = RESOURCE_DIR / "preprocessing_stats.json"

# every model family in ``models/`` is offered, one sub-folder per family
MODEL_SUFFIXES = (".joblib", ".pkl")
ROOT_FOLDER = "(models root)"

# CARTO basemaps API key for the folium map.  Never hard-coded: it is read from
# Streamlit secrets (``.streamlit/secrets.toml``, gitignored) or this variable.
CARTO_KEY_NAME = "CARTO_API_KEY"

REQUIRED_INPUTS = {
    "raw history": RAW_HISTORY,
    "prepared X": PREPARED_X,
    "prepared y": PREPARED_Y,
    "station coordinates": LOCATIONS_CSV,
    "preprocessing constants": PREPROCESSING_STATS,
}


def missing_inputs() -> dict[str, Path]:
    """Return the required files that are absent, for a startup diagnostic."""
    return {name: path for name, path in REQUIRED_INPUTS.items() if not path.exists()}


def discover_models() -> dict[str, list[Path]]:
    """Model files under the shared models directory, grouped by folder.

    Keys are folder paths relative to ``MODEL_DIR`` (``ROOT_FOLDER`` for files
    sitting directly in it); only folders holding at least one model are listed.
    """
    if not MODEL_DIR.is_dir():
        return {}
    groups: dict[str, list[Path]] = {}
    for path in sorted(MODEL_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES:
            parent = path.parent.relative_to(MODEL_DIR).as_posix()
            groups.setdefault(ROOT_FOLDER if parent == "." else parent, []).append(path)
    return dict(sorted(groups.items()))


def carto_api_key() -> str | None:
    """The CARTO basemaps key from Streamlit secrets, else the environment."""
    try:
        import streamlit as st

        key = st.secrets.get(CARTO_KEY_NAME)
    except Exception:  # no secrets.toml anywhere
        key = None
    return key or os.environ.get(CARTO_KEY_NAME) or None
