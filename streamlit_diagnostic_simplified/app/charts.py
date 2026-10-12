"""Altair charts.

Form first, colour last.  Curves over a shared x are lines; a per-station
comparison of one measure is bars; a metric against a coordinate is a scatter with
a trend line.  Colour then does one job per chart: model identity is categorical
(fixed slot order, never cycled), a station's magnitude is sequential (one hue,
light to dark), and a signed difference is diverging (two poles, neutral grey at
zero).

Every chart carries a tooltip, and every multi-series chart carries a legend, so
identity is never colour-alone.  The categorical palette's light-mode steps put
three hues below 3:1 against the surface, which obliges visible relief - the app
renders the same numbers as a table beside each chart.

Marks follow the house spec: 2px lines, >=8px points, recessive grid and axes,
values in text tokens rather than the series colour.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

import palette

POINT_SIZE = 70          # >= 8px across
LINE_WIDTH = 2
REFERENCE_DASH = [4, 4]
BAR_CORNER = 4
TOP_STATIONS = 25


def _dark() -> bool:
    try:
        import streamlit as st

        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return False


def apply_theme(chart: alt.Chart) -> alt.Chart:
    """Recessive grid and axes, text in text tokens, transparent surface."""
    dark = _dark()
    return (
        chart.configure_view(strokeWidth=0, fill=None)
        .configure_axis(
            labelColor=palette.text_secondary(dark),
            titleColor=palette.text_secondary(dark),
            gridColor=palette.grid(dark),
            domainColor=palette.grid(dark),
            tickColor=palette.grid(dark),
            labelFontSize=11,
            titleFontSize=12,
            grid=True,
        )
        .configure_legend(
            labelColor=palette.text_primary(dark),
            titleColor=palette.text_secondary(dark),
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_title(color=palette.text_primary(dark), fontSize=13, anchor="start")
    )


def _series_scale(names) -> alt.Scale:
    """Fixed slot order: dropping a model never repaints the survivors."""
    ordered = list(names)
    colours = palette.categorical(_dark())[: len(ordered)]
    return alt.Scale(domain=ordered, range=colours)


def _tidy(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [frame.assign(Model=name) for name, frame in frames.items()], ignore_index=True
    )


def _reference(data: pd.DataFrame, x: str, y: str) -> alt.Chart:
    dark = _dark()
    return (
        alt.Chart(data)
        .mark_line(strokeDash=REFERENCE_DASH, strokeWidth=1,
                   color=palette.text_secondary(dark), opacity=0.6)
        .encode(x=alt.X(f"{x}:Q"), y=alt.Y(f"{y}:Q"))
    )


def roc_chart(frames: dict[str, pd.DataFrame]) -> alt.Chart:
    data = _tidy(frames)
    names = list(frames)
    line = (
        alt.Chart(data)
        .mark_line(strokeWidth=LINE_WIDTH, interpolate="monotone")
        .encode(
            x=alt.X("fpr:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=_series_scale(names),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("Model:N"),
                     alt.Tooltip("fpr:Q", format=".3f", title="FPR"),
                     alt.Tooltip("tpr:Q", format=".3f", title="TPR")],
        )
    )
    diagonal = _reference(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]}), "fpr", "tpr")
    return (diagonal + line).properties(height=280)


def pr_chart(frames: dict[str, pd.DataFrame], base_rate: float) -> alt.Chart:
    data = _tidy(frames)
    names = list(frames)
    line = (
        alt.Chart(data)
        .mark_line(strokeWidth=LINE_WIDTH, interpolate="monotone")
        .encode(
            x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=_series_scale(names),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("Model:N"),
                     alt.Tooltip("recall:Q", format=".3f", title="Recall"),
                     alt.Tooltip("precision:Q", format=".3f", title="Precision")],
        )
    )
    floor = _reference(
        pd.DataFrame({"recall": [0, 1], "precision": [base_rate, base_rate]}),
        "recall", "precision",
    )
    return (floor + line).properties(height=280)


def calibration_chart(frames: dict[str, pd.DataFrame]) -> alt.Chart:
    data = _tidy(frames)
    names = list(frames)
    line = (
        alt.Chart(data)
        .mark_line(strokeWidth=LINE_WIDTH, point=alt.OverlayMarkDef(size=POINT_SIZE))
        .encode(
            x=alt.X("predicted:Q", title="Predicted probability",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("observed:Q", title="Observed rain rate",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=_series_scale(names),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("Model:N"),
                     alt.Tooltip("predicted:Q", format=".3f", title="Predicted"),
                     alt.Tooltip("observed:Q", format=".3f", title="Observed"),
                     alt.Tooltip("count:Q", format=",", title="Days")],
        )
    )
    perfect = _reference(
        pd.DataFrame({"predicted": [0, 1], "observed": [0, 1]}), "predicted", "observed"
    )
    return (perfect + line).properties(height=280)


def threshold_chart(sweep: pd.DataFrame, threshold: float) -> alt.Chart:
    """Three curves over one x - the first three slots validate all-pairs."""
    tidy = sweep.melt("threshold", var_name="Metric", value_name="Value")
    order = ["f1", "precision", "recall"]
    colours = palette.categorical(_dark())[:3]
    line = (
        alt.Chart(tidy)
        .mark_line(strokeWidth=LINE_WIDTH)
        .encode(
            x=alt.X("threshold:Q", title="Decision threshold"),
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", scale=alt.Scale(domain=order, range=colours),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("Metric:N"),
                     alt.Tooltip("threshold:Q", format=".2f"),
                     alt.Tooltip("Value:Q", format=".3f")],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(strokeDash=REFERENCE_DASH, strokeWidth=1,
                   color=palette.text_secondary(_dark()))
        .encode(x="threshold:Q",
                tooltip=[alt.Tooltip("threshold:Q", format=".2f", title="In use")])
    )
    return (line + marker).properties(height=280)


def station_bar_chart(
    frame: pd.DataFrame,
    metric: str,
    label: str,
    signed: bool,
    lower_is_better: bool,
) -> alt.Chart:
    """Worst-first ranking of one measure across stations."""
    ordered = frame.sort_values(metric, ascending=not lower_is_better).head(TOP_STATIONS)
    dark = _dark()

    if signed:
        # diverging: two poles with a neutral grey at zero, symmetric so the
        # midpoint really means "no difference"
        limit = float(ordered[metric].abs().max() or 1.0)
        scale = alt.Scale(
            domain=[-limit, 0.0, limit],
            range=[palette.DIVERGING[0], palette.neutral(dark), palette.DIVERGING[-1]],
        )
    else:
        scale = alt.Scale(range=[palette.SEQUENTIAL[2], palette.SEQUENTIAL[-2]])

    return (
        alt.Chart(ordered)
        .mark_bar(cornerRadiusEnd=BAR_CORNER, height={"band": 0.8})
        .encode(
            y=alt.Y("station:N", sort=list(ordered["station"]), title=None),
            x=alt.X(f"{metric}:Q", title=label),
            color=alt.Color(f"{metric}:Q", scale=scale, legend=None),
            tooltip=[alt.Tooltip("station:N", title="Station"),
                     alt.Tooltip(f"{metric}:Q", format=".3f", title=label),
                     alt.Tooltip("support:Q", format=",", title="Days"),
                     alt.Tooltip("positive_rate:Q", format=".3f", title="Rain rate")],
        )
        .properties(height=max(240, 18 * len(ordered)))
    )


def gradient_chart(frame: pd.DataFrame, metric: str, label: str, axis: str) -> alt.Chart:
    """The metric against a coordinate, with a trend line for the eye to follow."""
    title = "Longitude" if axis == "lon" else "Latitude"
    dark = _dark()
    base = alt.Chart(frame)

    position = {
        "x": alt.X(f"{axis}:Q", title=title, scale=alt.Scale(zero=False)),
        "y": alt.Y(f"{metric}:Q", title=label, scale=alt.Scale(zero=False)),
    }
    points = base.mark_circle(
        size=POINT_SIZE, opacity=0.85, color=palette.categorical(dark)[0]
    ).encode(
        **position,
        tooltip=[alt.Tooltip("station:N", title="Station"),
                 alt.Tooltip(f"{metric}:Q", format=".3f", title=label),
                 alt.Tooltip(f"{axis}:Q", format=".2f", title=title),
                 alt.Tooltip("support:Q", format=",", title="Days")],
    )
    # the trend is a reference mark, so it stays recessive rather than competing
    # with the data colour
    trend = (
        base.transform_regression(axis, metric)
        .mark_line(strokeWidth=LINE_WIDTH, color=palette.text_secondary(dark),
                   strokeDash=REFERENCE_DASH, opacity=0.7)
        .encode(**position)
    )
    return (points + trend).properties(height=260)


def prediction_chart(results: pd.DataFrame) -> alt.Chart:
    """One bar per model: its rain-tomorrow probability, ticked at its own threshold."""
    scored = results.dropna(subset=["P(rain tomorrow)"])
    order = list(scored["Model"])
    dark = _dark()
    tooltip = [alt.Tooltip("Model:N"),
               alt.Tooltip("P(rain tomorrow):Q", format=".3f"),
               alt.Tooltip("Threshold:Q", format=".2f"),
               alt.Tooltip("Prediction:N")]
    bars = (
        alt.Chart(scored)
        .mark_bar(cornerRadiusEnd=BAR_CORNER, height={"band": 0.7})
        .encode(
            y=alt.Y("Model:N", sort=order, title=None),
            x=alt.X("P(rain tomorrow):Q", title="P(rain tomorrow)",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=_series_scale(order), legend=None),
            tooltip=tooltip,
        )
    )
    ticks = (
        alt.Chart(scored)
        .mark_tick(thickness=2, size=22, color=palette.text_primary(dark))
        .encode(y=alt.Y("Model:N", sort=order), x="Threshold:Q", tooltip=tooltip)
    )
    return (bars + ticks).properties(height=max(120, 36 * len(order)))
