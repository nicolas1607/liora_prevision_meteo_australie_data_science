"""Rain-tomorrow model evaluation, with a geographic read on where it goes wrong.

Pick one or more EXP-007 bundles, point them at a validation set, and the app
scores it, reports the usual classification metrics, and breaks performance down
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
    "nan": "Pass NaN to XGBoost (degraded - not valid for bias claims)",
}

RAW_SOURCE = "Full weatherAUS history (raw, with stations)"
PREPARED_SOURCE = "Prepared test matrix (weatherAUS_X_test)"
UPLOAD_SOURCE = "Upload a file"


# --------------------------------------------------------------------- caching


@st.cache_resource(show_spinner=False)
def _load_model(path: str, label: str):
    return registry.load(registry.ModelEntry(key=label, label=label, path=Path(path)))


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
def _read_upload(payload: bytes, name: str) -> pd.DataFrame:
    import io

    return data_io.read_table(io.BytesIO(payload), name=name)


@st.cache_data(show_spinner=False)
def _predict(path: str, label: str, matrix: np.ndarray) -> np.ndarray:
    return _load_model(path, label).predict_proba(matrix)


@st.cache_data(show_spinner=False)
def _calibrate(path: str, label: str, proba: np.ndarray) -> np.ndarray:
    calibrator = _load_model(path, label).isotonic
    return np.clip(calibrator.predict(proba), 0.0, 1.0)


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


def _coverage_view(validation, entries, coverages, policy, location_column):
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
        "Every bundle expects the same 44-column contract: 23 same-day columns plus "
        "lag-1, 3-day-window and upwind-neighbour features."
    )

    summary = pd.DataFrame([
        {
            "Model": entry.label,
            "Present": len(cov.present),
            "Missing": len(cov.missing),
            "Scoreable": "yes" if cov.is_scoreable else "no",
        }
        for entry, cov in zip(entries, coverages)
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    first = coverages[0]
    if first.missing:
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
        "f1", "precision", "recall", "accuracy", "balanced_accuracy", "mcc",
        "roc_auc", "pr_auc", "brier", "fpr", "fnr",
        "positive_rate", "predicted_positive_rate", "support",
    ]
    table = table[[c for c in display_order if c in table.columns]]

    headline = table.iloc[0]
    cols = st.columns(5)
    for col, key in zip(cols, ["f1", "precision", "recall", "roc_auc", "pr_auc"]):
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
        fmap = mapping.build_comparison_map(station_frames, metric, mapping.TILES[tiles_label])
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
            tiles=mapping.TILES[tiles_label],
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
            file_name=f"per_station_{name.replace(' ', '_')}.csv",
            mime="text/csv",
        )


def _model_cards(entries, models):
    st.subheader("Model bundles")
    for entry, model in zip(entries, models):
        with st.expander(entry.label, expanded=False):
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
                    st.caption("This bundle carries no recorded metrics.")
                params = model.meta.get("params") or model.meta.get("best_params")
                if isinstance(params, dict):
                    st.caption("Hyper-parameters")
                    st.json({k: (float(v) if isinstance(v, np.floating) else v)
                             for k, v in params.items()}, expanded=False)
            st.caption(f"`{entry.path.name}`")


# ------------------------------------------------------------------------ main


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
        "Score the EXP-007 XGBoost bundles on a validation set, then break the result "
        "down by weather station to see whether the model's skill depends on where you are."
    )

    if not _require_inputs():
        return

    validation = _sidebar_data()
    if validation is None:
        st.info("Choose a validation source in the sidebar to begin.")
        with st.expander("What the app expects"):
            st.markdown(_EXPECTATIONS)
            st.dataframe(load_stations()[["Location", "lat", "lon"]],
                         use_container_width=True, hide_index=True)
        return

    location_column = _sidebar_location(validation)

    st.sidebar.header("3 · Models")
    entries = registry.discover()
    if not entries:
        st.error(
            f"No EXP-007 bundles (`{paths.MODEL_GLOB}`) found in "
            f"`{paths.MODEL_DIR}`. Set WEATHER_MODEL_DIR if they live elsewhere."
        )
        return

    scorers = [e for e in entries if not registry.is_calibrator(_load_model(str(e.path), e.label))]
    calibrators = [e for e in entries if e not in scorers]

    # Default to the production bundle rather than a research one: the
    # leave-stations-out models score optimistically here, because this data covers
    # the very stations they were trained on.
    preferred = next((e for e in scorers if "production" in e.path.stem), None)
    picked_labels = st.sidebar.multiselect(
        "Compare",
        [e.label for e in scorers],
        default=[(preferred or scorers[-1]).label] if scorers else [],
        max_selections=8,  # the categorical palette defines exactly eight slots
    )
    picked = [e for e in scorers if e.label in picked_labels]

    st.sidebar.header("4 · Decision rule")
    threshold_mode = st.sidebar.radio(
        "Threshold", ["Each bundle's own", "Single override"], label_visibility="collapsed"
    )
    override = None
    if threshold_mode == "Single override":
        override = st.sidebar.slider("Threshold", 0.0, 1.0, 0.50, 0.01)

    calibrator_entry = None
    if calibrators:
        if st.sidebar.checkbox(
            f"Apply {calibrators[0].label}",
            value=False,
            help="Maps raw scores onto calibrated probabilities before thresholding.",
        ):
            calibrator_entry = calibrators[0]

    st.sidebar.header("5 · Missing features")
    policy_key = st.sidebar.radio(
        "Policy",
        list(MISSING_POLICY),
        format_func=lambda k: MISSING_POLICY[k],
        label_visibility="collapsed",
    )

    if not picked:
        st.info("Select at least one model in the sidebar.")
        return

    models = [_load_model(str(e.path), e.label) for e in picked]

    coverages = [data_io.assess_coverage(validation.frame, m.features) for m in models]
    frame = validation.frame

    tabs = st.tabs([
        "Data & coverage", "Metrics", "Bias map", "Bias diagnostics", "Model cards"
    ])

    with tabs[0]:
        _coverage_view(validation, picked, coverages, policy_key, location_column)
    with tabs[4]:
        _model_cards(picked, models)

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
            "**Degraded scoring is on.** Missing features are being passed to XGBoost as "
            "NaN, so every tree sends those rows down its default branch. The numbers "
            "below are not a valid measurement of these models and must not be used to "
            "draw conclusions about location bias.",
            icon="⚠️",
        )

    y_true = pd.to_numeric(validation.target, errors="coerce").to_numpy()
    valid_rows = ~np.isnan(y_true)

    scores: dict[str, Scored] = {}
    for entry, model, _cov in usable:
        matrix = data_io.build_matrix(frame, model.features)
        proba = _predict(str(entry.path), entry.label, matrix)
        calibrated = False
        if calibrator_entry is not None:
            proba = _calibrate(str(calibrator_entry.path), calibrator_entry.label, proba)
            calibrated = True
        threshold = override if override is not None else model.threshold
        scores[entry.label] = Scored(
            model=entry.label,
            y_true=y_true[valid_rows].astype(int),
            proba=proba[valid_rows],
            threshold=float(threshold),
            calibrated=calibrated,
        )

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
**A scoreable validation file needs all 44 features the bundles were trained on** —
23 same-day columns plus lag-1, 3-day-window and upwind-neighbour features. The
**Data & coverage** tab lists exactly which ones your file is missing.

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
