"""Rain-tomorrow model evaluation, with a geographic read on where it goes wrong.

Pick a model folder under ``models/``, then one or more of its ``.joblib`` /
``.pkl`` files, point them at a validation set, and the app scores it, reports the usual classification metrics, and breaks performance down
by weather station on a folium map so location bias is visible rather than
averaged away.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402  - shared data/model locations

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

import charts  # noqa: E402
import data_io  # noqa: E402
import evaluation  # noqa: E402
import mapping  # noqa: E402
import prediction  # noqa: E402
import model_registry as registry  # noqa: E402
from evaluation import STATION_METRICS, Scored  # noqa: E402
from locations import attach_coordinates, load_stations  # noqa: E402

st.set_page_config(
    page_title="AUS rain model evaluation",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MISSING_POLICY = {
    "block": "Block the model (safest)",
    "nan": "Pass NaN to the model (degraded, tree models only - not valid for bias claims)",
}

MAX_MODELS = 8  # the categorical palette defines exactly eight slots
LOADED_KEY = "loaded_models"  # session state: entry keys in the comparison, in order

RAW_SOURCE = "Full weatherAUS history (raw, with stations)"
PREPARED_SOURCE = "Prepared test matrix (weatherAUS_X_test)"
UPLOAD_SOURCE = "Upload a file"


# --------------------------------------------------------------------- caching


@st.cache_resource(show_spinner=False)
def _load_model(path: str, folder: str):
    entry = registry.ModelEntry(
        key=f"{folder}/{Path(path).name}", label=Path(path).name, path=Path(path), folder=folder
    )
    return registry.load(entry)


@st.cache_data(show_spinner=False)
def _read_prepared() -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = data_io.bundled_prepared()
    if pair is None:
        raise FileNotFoundError("the prepared weatherAUS test split is not in data/")
    return pd.read_csv(pair[0]), pd.read_csv(pair[1])


@st.cache_data(show_spinner="Applying the training transform to the raw history...")
def _read_raw() -> data_io.ValidationData:
    path = data_io.bundled_raw()
    if path is None:
        raise FileNotFoundError("the raw weatherAUS table is not in data/")
    frame = pd.read_csv(path, parse_dates=["Date"])
    return data_io.build_validation_data(frame, path.name)


@st.cache_data(show_spinner=False)
def _raw_history() -> pd.DataFrame:
    return pd.read_csv(paths.RAW_HISTORY, parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def _pre_imputation_frame() -> pd.DataFrame | None:
    return data_io._pre_imputation()


@st.cache_data(show_spinner="Processing the input...")
def _process_input(location: str, date, values: tuple) -> tuple[pd.DataFrame, dict]:
    return prediction.build_input_row(
        location, date, dict(values), _raw_history(), _pre_imputation_frame()
    )


@st.cache_data(show_spinner=False)
def _read_upload(payload: bytes, name: str) -> pd.DataFrame:
    import io

    return data_io.read_table(io.BytesIO(payload), name=name)


@st.cache_data(show_spinner=False)
def _predict(path: str, folder: str, inputs: pd.DataFrame) -> np.ndarray:
    return _load_model(path, folder).predict_proba(inputs)


# ------------------------------------------------------------------ data input


def _date_window(validation: data_io.ValidationData) -> data_io.ValidationData:
    """Let the user evaluate a slice while features stay built on the full history."""
    span = validation.date_span
    if span is None:
        return validation
    low, high = span[0].date(), span[1].date()
    if low >= high:
        return validation

    default_start = max(low, pd.Timestamp("2016-01-01").date())
    picked = st.sidebar.date_input(
        "Evaluate dates",
        value=(default_start, high),
        min_value=low,
        max_value=high,
        help=(
            "Features are built over the whole history first, then sliced — so lag "
            "and 3-day terms at the start of the window still see the days before it."
        ),
    )
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        return validation
    start, end = picked
    sliced = validation.restrict_to_dates(start, end)
    st.sidebar.caption(f"{sliced.n_rows:,} of {validation.n_rows:,} rows in range.")
    return sliced


def _sidebar_data() -> data_io.ValidationData | None:
    st.sidebar.header("1 · Validation data")
    options = []
    if data_io.bundled_raw() is not None:
        options.append(RAW_SOURCE)
    if data_io.bundled_prepared() is not None:
        options.append(PREPARED_SOURCE)
    options.append(UPLOAD_SOURCE)
    choice = st.sidebar.radio("Source", options, label_visibility="collapsed")

    if choice == RAW_SOURCE:
        return _date_window(_read_raw())

    if choice == PREPARED_SOURCE:
        features, labels = _read_prepared()
        return data_io.build_validation_data(features, "weatherAUS_X_test.csv", labels)

    uploaded = st.sidebar.file_uploader(
        "Feature file (.csv / .parquet)", type=["csv", "parquet", "pq"]
    )
    if uploaded is None:
        return None
    frame = _read_upload(uploaded.getvalue(), uploaded.name)

    labels_frame = None
    label_file = st.sidebar.file_uploader(
        "Labels file (optional)",
        type=["csv", "parquet", "pq"],
        help="Only needed when the target is not a column of the feature file.",
    )
    if label_file is not None:
        labels_frame = _read_upload(label_file.getvalue(), label_file.name)

    target_column = None
    if labels_frame is None:
        candidates = [c for c in frame.columns]
        guess = data_io._first_present(frame, data_io.TARGET_CANDIDATES)
        target_column = st.sidebar.selectbox(
            "Target column",
            ["(none - predictions only)"] + candidates,
            index=(candidates.index(guess) + 1) if guess else 0,
        )
        if target_column.startswith("(none"):
            target_column = None

    return _date_window(
        data_io.build_validation_data(frame, uploaded.name, labels_frame, target_column)
    )


def _sidebar_location(validation: data_io.ValidationData) -> str | None:
    st.sidebar.header("2 · Station column")
    text_columns = [
        c for c in validation.frame.columns
        if validation.frame[c].dtype == object or c == validation.location_column
    ]
    if not text_columns:
        st.sidebar.info("No station column found - the map needs one.")
        return None
    default = 0
    if validation.location_column in text_columns:
        default = text_columns.index(validation.location_column) + 1
    picked = st.sidebar.selectbox("Groups the map by", ["(none)"] + text_columns, index=default)
    return None if picked == "(none)" else picked


# ----------------------------------------------------------------------- views


def _coverage_view(validation, entries, coverages, policy, location_column, names):
    st.subheader("Validation set")
    left, mid, right = st.columns(3)
    left.metric("Rows", f"{validation.n_rows:,}")
    mid.metric("Columns", f"{validation.frame.shape[1]:,}")
    if validation.target is not None:
        rate = float(pd.to_numeric(validation.target, errors="coerce").mean())
        right.metric("Rain-tomorrow rate", f"{rate:.1%}")
    else:
        right.metric("Rain-tomorrow rate", "no labels")

    st.caption(f"Source: `{validation.source}`")
    for note in validation.notes:
        st.caption(note)

    if location_column:
        joined, unmatched = attach_coordinates(validation.frame, location_column)
        matched = int(joined["lat"].notna().sum())
        st.write(
            f"**Stations** — {joined[location_column].nunique()} distinct values, "
            f"{matched:,} of {len(joined):,} rows matched to coordinates."
        )
        if unmatched:
            st.warning(
                f"{len(unmatched)} station name(s) are not in `resources/locations.csv` and will "
                f"be dropped from the map: {', '.join(unmatched[:12])}"
                + (" ..." if len(unmatched) > 12 else "")
            )
    else:
        st.warning(
            "No station column is selected, so the bias map and the per-station tabs "
            "stay empty. Pick one in the sidebar if your data has one."
        )

    st.divider()
    st.subheader("Feature coverage")
    st.caption(
        "Most model families expect the 23 base columns of the pipeline's scaled "
        "`X_train`; the EXP-007 XGBoost bundles add 21 lag-1, 3-day-window and "
        "upwind-neighbour features on top (44 in all)."
    )

    summary = pd.DataFrame([
        {
            "Model": names[entry.key],
            "Folder": entry.folder,
            "Expected": len(cov.present) + len(cov.missing),
            "Present": len(cov.present),
            "Missing": len(cov.missing),
            "Scoreable": "yes" if cov.is_scoreable else "no",
        }
        for entry, cov in zip(entries, coverages)
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    first = next((cov for cov in coverages if cov.missing), None)
    if first is not None:
        with st.expander(f"Missing columns ({len(first.missing)})", expanded=True):
            st.code(", ".join(first.missing), language=None)
            st.markdown(
                "These are engineered features. They **cannot** be rebuilt from an "
                "already-scaled matrix — the lag, window and neighbour terms live on "
                "the raw scale, so reconstructing them from scaled inputs would put "
                "them on the wrong scale entirely.\n\n"
                "Supply raw weatherAUS columns instead (`Date`, `Location` and the "
                "original observations) and the app will build all 44 features itself."
            )
            if policy == "block":
                st.info(
                    "Switch the **Missing features** policy in the sidebar to score "
                    "anyway — but see the warning it raises before trusting the result."
                )

    with st.expander("Preview the data"):
        st.dataframe(validation.frame.head(200), use_container_width=True)


def _metrics_view(scores: dict[str, Scored], validation):
    st.subheader("Overall metrics")
    rows = {name: evaluation.global_metrics(scored) for name, scored in scores.items()}
    table = pd.DataFrame(rows).T
    display_order = [
        "f1", "precision", "recall", "accuracy", "balanced_accuracy", "roc_auc",
        "positive_rate", "predicted_positive_rate", "support",
    ]
    table = table[[c for c in display_order if c in table.columns]]

    headline = table.iloc[0]
    cols = st.columns(4)
    for col, key in zip(cols, ["f1", "precision", "recall", "roc_auc"]):
        col.metric(key.replace("_", " ").upper(), f"{headline[key]:.4f}")
    st.caption(f"Headline figures are for **{table.index[0]}**.")

    st.dataframe(
        table.style.format({c: "{:.4f}" for c in table.columns if c != "support"})
        .format({"support": "{:,.0f}"}),
        use_container_width=True,
    )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("**ROC curve** — dashed line is a coin flip")
        st.altair_chart(
            charts.apply_theme(
                charts.roc_chart({n: evaluation.roc_points(s) for n, s in scores.items()})
            ),
            use_container_width=True,
        )
    with right:
        base_rate = float(np.mean(next(iter(scores.values())).y_true))
        st.markdown("**Precision-recall curve** — dashed line is the base rain rate")
        st.altair_chart(
            charts.apply_theme(
                charts.pr_chart(
                    {n: evaluation.pr_points(s) for n, s in scores.items()}, base_rate
                )
            ),
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Calibration** — how close predicted probabilities are to reality")
        st.caption(
            "Models scored through a sigmoid of their decision function (LinearSVC) "
            "output a ranking, not a probability, so their curve here is not meaningful."
        )
        st.altair_chart(
            charts.apply_theme(
                charts.calibration_chart(
                    {n: evaluation.calibration_bins(s) for n, s in scores.items()}
                )
            ),
            use_container_width=True,
        )
    with right:
        focus = st.selectbox("Threshold sweep for", list(scores), key="sweep_model")
        st.altair_chart(
            charts.apply_theme(
                charts.threshold_chart(
                    evaluation.threshold_sweep(scores[focus]), scores[focus].threshold
                )
            ),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Confusion matrices")
    for chunk in [list(scores)[i:i + 3] for i in range(0, len(scores), 3)]:
        for col, name in zip(st.columns(len(chunk)), chunk):
            with col:
                st.caption(f"{name} @ {scores[name].threshold:.2f}")
                st.dataframe(evaluation.confusion_frame(scores[name]),
                             use_container_width=True)


def _map_view(station_frames: dict[str, pd.DataFrame]):
    st.subheader("Where the model is strong and where it is weak")
    controls = st.columns([2, 2, 1, 1])
    metric = controls[0].selectbox(
        "Colour stations by",
        list(STATION_METRICS),
        index=list(STATION_METRICS).index("f1"),
        format_func=lambda k: STATION_METRICS[k]["label"],
    )
    tiles_label = controls[1].selectbox("Basemap", list(mapping.TILES))
    carto_key = paths.carto_api_key()
    tiles = mapping.tile_options(tiles_label, carto_key)
    if not carto_key and mapping.TILES[tiles_label] is not None:
        st.caption(
            f"No `{paths.CARTO_KEY_NAME}` configured, so the CARTO basemap falls back to "
            "OpenStreetMap. Add it to `.streamlit/secrets.toml` (see README)."
        )
    show_labels = controls[2].checkbox("Names", value=False)
    show_heat = controls[3].checkbox("Heat", value=False)

    names = list(station_frames)
    if len(names) > 1:
        mode = st.radio(
            "View",
            ["Single model", "Compare models (layer toggle)"],
            horizontal=True,
            label_visibility="collapsed",
        )
    else:
        mode = "Single model"

    if mode.startswith("Compare"):
        fmap = mapping.build_comparison_map(station_frames, metric, tiles)
        st.caption(
            "One layer per model on a shared colour scale — toggle them in the "
            "layer control to see which stations move."
        )
    else:
        chosen = st.selectbox("Model", names, key="map_model") if len(names) > 1 else names[0]
        frame = station_frames[chosen]
        fmap = mapping.build_map(
            frame,
            metric,
            tiles=tiles,
            show_labels=show_labels,
            show_heat=show_heat,
        )
        st.caption(
            "Circle **colour** is the selected metric, **size** is how many validation "
            "days that station contributed. Dashed rings mark stations whose error rate "
            "is more than 2 standard errors from the overall rate. Click a circle for "
            "its full breakdown."
        )

    st_folium(
        fmap,
        use_container_width=True,
        height=580,
        returned_objects=[],
        key=f"map-{metric}-{mode}-{tiles_label}",
    )


def _bias_view(scores: dict[str, Scored], station_frames: dict[str, pd.DataFrame], stations_by_row):
    st.subheader("Is the model's skill location-dependent?")
    name = st.selectbox("Model", list(station_frames), key="bias_model")
    frame = station_frames[name]
    scored = scores[name]

    test = evaluation.location_dependence_test(scored, stations_by_row)
    if test:
        cols = st.columns(4)
        cols[0].metric("Chi-square", f"{test['chi2']:,.1f}")
        cols[1].metric("p-value", f"{test['p_value']:.2e}")
        cols[2].metric("Cramer's V", f"{test['cramers_v']:.3f}")
        cols[3].metric("Stations", f"{len(frame):,}")
        verdict = (
            "Error rate **does** vary by station beyond chance"
            if test["p_value"] < 0.05
            else "No statistically significant variation in error rate by station"
        )
        strength = (
            "negligible" if test["cramers_v"] < 0.1
            else "small" if test["cramers_v"] < 0.2
            else "moderate" if test["cramers_v"] < 0.3
            else "large"
        )
        st.markdown(
            f"{verdict} (p = {test['p_value']:.3g}). Cramer's V puts the effect size at "
            f"**{strength}** — a large sample can make a tiny difference significant, so "
            f"read the two together."
        )

    metric = st.selectbox(
        "Break down by",
        list(STATION_METRICS),
        index=list(STATION_METRICS).index("f1"),
        format_func=lambda k: STATION_METRICS[k]["label"],
        key="bias_metric",
    )
    spec = STATION_METRICS[metric]
    signed = evaluation.is_signed_metric(metric)

    spread = evaluation.disparity_summary(frame, metric)
    if spread:
        cols = st.columns(5)
        cols[0].metric("Best station", f"{spread['best']:.3f}")
        cols[1].metric("Worst station", f"{spread['worst']:.3f}")
        cols[2].metric("Spread", f"{spread['spread']:.3f}")
        cols[3].metric("P10 – P90", f"{spread['p10']:.3f} – {spread['p90']:.3f}")
        cols[4].metric("Std dev", f"{spread['std']:.3f}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Stations ranked worst-first by {spec['label'].lower()}** (top 25)")
        st.altair_chart(
            charts.apply_theme(
                charts.station_bar_chart(
                    frame, metric, spec["label"], signed, spec["lower_is_better"]
                )
            ),
            use_container_width=True,
        )
    with right:
        axis = st.radio("Geographic axis", ["lon", "lat"], horizontal=True,
                        format_func=lambda a: "Longitude" if a == "lon" else "Latitude")
        st.altair_chart(
            charts.apply_theme(charts.gradient_chart(frame, metric, spec["label"], axis)),
            use_container_width=True,
        )
        gradient = evaluation.spatial_gradient(frame, metric)
        if gradient:
            st.caption(
                f"Correlation with longitude r = {gradient['lon_pearson']:+.2f} "
                f"(Spearman {gradient['lon_spearman']:+.2f}); with latitude "
                f"r = {gradient['lat_pearson']:+.2f} "
                f"(Spearman {gradient['lat_spearman']:+.2f}). A strong value means the "
                f"model is systematically weaker on one side of the continent."
            )

    st.divider()
    flagged = frame[frame["error_z"].abs() >= 2].sort_values("error_z", ascending=False)
    st.markdown(f"**Stations out of line with the overall error rate** ({len(flagged)})")
    if flagged.empty:
        st.success("No station's error rate sits more than 2 standard errors from the overall rate.")
    else:
        st.dataframe(
            flagged[["station", "support", "positive_rate", "error_rate",
                     "error_rate_delta", "error_z", "precision", "recall", "f1"]]
            .style.format({
                "support": "{:,.0f}", "positive_rate": "{:.3f}", "error_rate": "{:.3f}",
                "error_rate_delta": "{:+.3f}", "error_z": "{:+.2f}",
                "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Full per-station table"):
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download as CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name=f"per_station_{Path(name).stem}.csv",
            mime="text/csv",
        )


def _model_cards(entries, models):
    st.subheader("Model files")
    for entry, model in zip(entries, models):
        with st.expander(f"{entry.folder}/{entry.label}", expanded=False):
            left, right = st.columns([2, 3])
            with left:
                st.dataframe(
                    pd.DataFrame(registry.describe(model).items(), columns=["Field", "Value"]),
                    use_container_width=True,
                    hide_index=True,
                )
            with right:
                reported = registry.reported_metrics(model)
                if reported:
                    st.caption("Metrics recorded by the training notebook")
                    st.dataframe(
                        pd.DataFrame(reported.items(), columns=["Metric", "Value"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("This file carries no recorded metrics.")
                params = registry.hyper_parameters(model)
                if isinstance(params, dict):
                    st.caption("Hyper-parameters")
                    st.json({k: (float(v) if isinstance(v, np.floating) else v)
                             for k, v in params.items()}, expanded=False)


def _prediction_view(entries, models, names):
    st.subheader("Predict rain tomorrow from raw observations")
    st.caption(
        "Enter the original, unscaled observations for one station and day. The app "
        "applies the training transform - scaling, wind-direction and date encodings, "
        "and the lag, 3-day-window and upwind-neighbour features built from the year of "
        "history before that date - then asks every model in the comparison. Leave a "
        "field blank if it was not observed: it stays missing."
    )
    history = _raw_history()
    catalogue = set(load_stations()["Location"])
    stations = sorted(set(history["Location"].unique()) & catalogue)
    history_end = history["Date"].max().date()

    with st.form("prediction_input"):
        left, right = st.columns(2)
        location = left.selectbox("Station", stations, key="pred::Location")
        date = right.date_input(
            "Observation date", value=history_end, key="pred::Date",
            help=f"Lags and neighbours come from the history, which ends on {history_end}.",
        )
        values: dict = {}
        for group, fields in prediction.RAW_FIELDS.items():
            st.markdown(f"**{group}**")
            columns = st.columns(4)
            for column, (name, (label, unit, low, high, step)) in zip(
                columns * 2, fields.items()
            ):
                values[name] = column.number_input(
                    f"{label} ({unit})", min_value=low, max_value=high, value=None,
                    step=step, key=f"pred::{name}", placeholder="blank = missing",
                )
        st.markdown("**Wind direction**")
        columns = st.columns(4)
        for column, (name, label) in zip(columns, prediction.DIRECTION_FIELDS.items()):
            values[name] = column.selectbox(
                label, [None] + prediction.COMPASS, key=f"pred::{name}",
                format_func=lambda v: "(missing)" if v is None else v,
            )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        st.session_state["prediction_request"] = (location, date, tuple(values.items()))
    request = st.session_state.get("prediction_request")
    if request is None:
        st.info("Fill in what you know and click **Predict**.")
        return

    location, date, values = request
    features, context = _process_input(location, date, values)
    entered = sum(v is not None for _, v in values)
    rain_today = context["rain_today"]
    notes = [
        f"**{location}, {pd.Timestamp(date).date()}** - {entered} of "
        f"{len(values)} raw fields entered; RainToday = "
        + ("unknown (Rainfall blank)" if pd.isna(rain_today)
           else "Yes" if rain_today else "No") + " (Rainfall > 1 mm).",
    ]
    if context["previous_day"]:
        notes.append(f"Lag and 3-day features use {context['history_days']} days of history "
                     "for this station.")
    else:
        notes.append(
            f"No history for {location} on the previous day (the history ends on "
            f"{context['history_end'].date()}), so the lag and 3-day features are missing."
        )
    if context["upwind"]:
        notes.append(
            f"Upwind neighbours {', '.join(context['upwind'])}: "
            f"{len(context['upwind_with_data'])} have observations that day."
        )
    for note in notes:
        st.caption(note)

    results = prediction.predict(entries, models, names, features)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            results.style.format({"P(rain tomorrow)": "{:.3f}", "Threshold": "{:.2f}"},
                                 na_rep="-"),
            use_container_width=True, hide_index=True,
        )
    with right:
        if results["P(rain tomorrow)"].notna().any():
            st.altair_chart(charts.apply_theme(charts.prediction_chart(results)),
                            use_container_width=True)
            st.caption("Bar = probability, tick = that model's decision threshold.")
    blocked = results[results["Status"] != "ok"]
    if not blocked.empty:
        st.warning(
            f"{len(blocked)} model(s) could not score this input: linear models "
            "(LogisticRegression, LinearSVC) cannot take missing values. Fill in the "
            "listed fields to include them."
        )

    with st.expander("Processed input - what the models receive"):
        table = prediction.processed_table(features, models)
        st.dataframe(table.style.format({"Value": "{:.4f}"}, na_rep="missing"),
                     use_container_width=True, hide_index=True)
        st.caption(
            "Base weather columns are on the training scale (RobustScaler); engineered "
            "columns are on the raw scale, as in training. Bundles with their own "
            "StandardScaler or interaction terms apply them on top."
        )
        st.download_button(
            "Download as CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name=f"processed_{location}_{pd.Timestamp(date).date()}.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------------------ main


def _display_names(entries) -> dict[str, str]:
    """File name per model, prefixed with its folder only when two would clash."""
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.label] = counts.get(entry.label, 0) + 1
    return {
        e.key: e.label if counts[e.label] == 1 else f"{e.folder}/{e.label}" for e in entries
    }


def _require_inputs() -> bool:
    """Stop with an actionable message when the shared data is not in place."""
    missing = paths.missing_inputs()
    if not missing:
        return True
    st.error("Required files are missing - the project data is not in this checkout.")
    bullets = [f"- **{name}**: `{path}`" for name, path in missing.items()]
    bullets.append("")
    bullets.append(
        f"Copy the Google Drive `/_data` files into `{paths.DATA_DIR}`, or point "
        "`WEATHER_DATA_DIR` / `WEATHER_MODEL_DIR` at where they live."
    )
    st.markdown(chr(10).join(bullets))
    return False


def main() -> None:
    st.title("🌧️ Rain-tomorrow model evaluation")
    st.caption(
        "Score any model saved under `models/` on a validation set, then break the result "
        "down by weather station to see whether the model's skill depends on where you are."
    )

    if not _require_inputs():
        return

    validation = _sidebar_data()
    if validation is None:
        st.info(
            "Choose a validation source in the sidebar to evaluate the models. The "
            "**Prediction** tab works without one."
        )
        with st.expander("What the app expects"):
            st.markdown(_EXPECTATIONS)
            st.dataframe(load_stations()[["Location", "lat", "lon"]],
                         use_container_width=True, hide_index=True)

    location_column = _sidebar_location(validation) if validation is not None else None

    st.sidebar.header("3 · Models")
    folders = registry.discover()
    if not folders:
        st.error(
            f"No model files ({', '.join(paths.MODEL_SUFFIXES)}) found in "
            f"`{paths.MODEL_DIR}`. Set WEATHER_MODEL_DIR if they live elsewhere."
        )
        return

    folder = st.sidebar.selectbox(
        "Folder",
        list(folders),
        format_func=lambda f: f"{f} ({len(folders[f])})",
        help=f"Sub-folders of `{paths.MODEL_DIR}` that hold .joblib or .pkl files.",
    )
    entries = folders[folder]
    by_key = {e.key: e for group in folders.values() for e in group}

    # Ticking a box does nothing on its own: models are loaded only when the
    # button is clicked, and each click *adds* to the comparison, so families
    # from different folders (LogisticRegression, XGBoost ...) can be stacked.
    loaded = [k for k in st.session_state.get(LOADED_KEY, []) if k in by_key]
    if st.session_state.pop("_clear_picks", False):
        for key in [k for k in st.session_state if str(k).startswith("pick::")]:
            del st.session_state[key]

    with st.sidebar.form("model_picker", border=False):
        st.caption("Tick model files, then click **Add to comparison**. Repeat in other folders to stack them.")
        ticked = []
        for entry in entries:
            if entry.key in loaded:
                st.checkbox(f"{entry.label} (loaded)", value=True, disabled=True,
                            key=f"done::{entry.key}")
            elif st.checkbox(entry.label, key=f"pick::{entry.key}"):
                ticked.append(entry.key)
        submitted = st.form_submit_button(
            "Add to comparison", type="primary", use_container_width=True
        )
    if submitted and ticked:
        room = MAX_MODELS - len(loaded)
        if len(ticked) > room:
            st.session_state["_skipped"] = len(ticked) - max(room, 0)
        st.session_state[LOADED_KEY] = loaded + ticked[:max(room, 0)]
        st.session_state["_clear_picks"] = True
        st.rerun()
    skipped = st.session_state.pop("_skipped", 0)
    if skipped:
        st.sidebar.warning(
            f"At most {MAX_MODELS} models can be compared - {skipped} not added. "
            "Remove one below to make room."
        )

    picked, models = [], []
    for entry in (by_key[k] for k in loaded):
        try:
            with st.spinner(f"Loading {entry.label}..."):
                model = _load_model(str(entry.path), entry.folder)
        except Exception as error:  # a missing library or an unreadable pickle
            st.sidebar.error(f"`{entry.label}` could not be loaded: {type(error).__name__}: {error}")
            continue
        if not registry.can_score(model):
            st.sidebar.warning(f"`{entry.label}` is not a classifier this app can score - skipped.")
            continue
        picked.append(entry)
        models.append(model)

    if loaded:
        st.sidebar.markdown(f"**In the comparison** ({len(loaded)}/{MAX_MODELS})")
        for key in loaded:
            entry = by_key[key]
            name_col, remove_col = st.sidebar.columns([5, 1], vertical_alignment="center")
            name_col.caption(f"`{entry.folder}` / {entry.label}")
            if remove_col.button("✕", key=f"remove::{key}", help=f"Remove {entry.label}"):
                st.session_state[LOADED_KEY] = [k for k in loaded if k != key]
                st.rerun()
        if st.sidebar.button("Remove all", use_container_width=True):
            st.session_state[LOADED_KEY] = []
            st.rerun()

    st.sidebar.header("4 · Missing features")
    policy_key = st.sidebar.radio(
        "Policy",
        list(MISSING_POLICY),
        format_func=lambda k: MISSING_POLICY[k],
        label_visibility="collapsed",
    )

    if not picked:
        st.info(
            "Pick a folder, tick one or more model files and click **Add to comparison**. "
            "Repeat with other folders to compare model families side by side."
        )
        return

    names = _display_names(picked)
    tabs = st.tabs([
        "Data & coverage", "Metrics", "Bias map", "Bias diagnostics", "Model cards",
        "Prediction",
    ])

    with tabs[4]:
        _model_cards(picked, models)
    with tabs[5]:
        _prediction_view(picked, models, names)

    if validation is None:
        for tab in tabs[:4]:
            with tab:
                st.info("Choose a validation source in the sidebar to fill this tab.")
        return

    coverages = [data_io.assess_coverage(validation.frame, m.input_features) for m in models]
    frame = validation.frame
    with tabs[0]:
        _coverage_view(validation, picked, coverages, policy_key, location_column, names)

    if validation.target is None:
        for tab, note in zip(tabs[1:4], ["Metrics", "The bias map", "Bias diagnostics"]):
            with tab:
                st.info(f"{note} need a target column. Point the sidebar at one.")
        return

    usable = [
        (entry, model, cov)
        for entry, model, cov in zip(picked, models, coverages)
        if cov.is_scoreable or policy_key == "nan"
    ]
    if not usable:
        for tab in tabs[1:4]:
            with tab:
                st.error(
                    "None of the selected models can be scored on this data — see "
                    "**Data & coverage** for the columns that are missing."
                )
        return

    if policy_key == "nan" and any(not cov.is_scoreable for _, _, cov in usable):
        st.warning(
            "**Degraded scoring is on.** Missing features are being passed to the model as "
            "NaN, so every tree sends those rows down its default branch. The numbers "
            "below are not a valid measurement of these models and must not be used to "
            "draw conclusions about location bias.",
            icon="⚠️",
        )

    y_true = pd.to_numeric(validation.target, errors="coerce").to_numpy()
    valid_rows = ~np.isnan(y_true)

    scores: dict[str, Scored] = {}
    for entry, model, _cov in usable:
        inputs = data_io.build_frame(frame, model.input_features)
        try:
            proba = _predict(str(entry.path), entry.folder, inputs)
        except Exception as error:  # e.g. a linear model handed NaN under the degraded policy
            st.error(f"`{entry.label}` could not be scored: {type(error).__name__}: {error}")
            continue
        scores[names[entry.key]] = Scored(
            model=names[entry.key],
            y_true=y_true[valid_rows].astype(int),
            proba=proba[valid_rows],
            threshold=float(model.threshold),
        )

    if not scores:
        return

    with tabs[1]:
        _metrics_view(scores, validation)

    if not location_column:
        for tab in tabs[2:4]:
            with tab:
                st.info("Pick a station column in the sidebar to unlock the geographic views.")
        return

    joined, _ = attach_coordinates(frame, location_column)
    stations_by_row = joined.loc[valid_rows, "station"].reset_index(drop=True)
    coords = joined[["station", "lat", "lon"]].drop_duplicates("station")

    min_support = st.sidebar.number_input(
        "Min days per station", min_value=1, max_value=2000, value=30, step=10,
        help="Stations with fewer validation days are left off the map and the tables.",
    )

    station_frames = {}
    for name, scored in scores.items():
        per_station = evaluation.per_station_metrics(scored, stations_by_row, int(min_support))
        if not per_station.empty:
            station_frames[name] = per_station.merge(coords, on="station", how="left")

    if not station_frames:
        for tab in tabs[2:4]:
            with tab:
                st.warning("No station clears the minimum-days filter. Lower it in the sidebar.")
        return

    with tabs[2]:
        _map_view(station_frames)
    with tabs[3]:
        _bias_view(scores, station_frames, stations_by_row)


_EXPECTATIONS = """
**A scoreable validation file needs every feature the selected model was trained on** —
the 23 base columns for most families, 44 for the EXP-007 XGBoost bundles (23 plus
lag-1, 3-day-window and upwind-neighbour features). The **Data & coverage** tab lists
exactly which ones your file is missing.

To get the geographic views as well, include:

- a **station column** (`Location`) whose values match the catalogue below, and
- a **target column** (`RainTomorrow`, as `0/1` or `No/Yes`) — either in the same
  file or as a separate labels file with one row per feature row.

A `Date` column is optional, but with `Date` + `Location` the app can rebuild the
lag, rolling-window and neighbour features for you (sidebar → *Missing features*).
Rebuilt values assume your inputs are already on the same scale as training.
"""


if __name__ == "__main__":
    main()
