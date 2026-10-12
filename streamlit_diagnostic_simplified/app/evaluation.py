"""Metrics, per-station breakdowns and the location-bias statistics.

The per-station work is the point of the app.  A single F1 averages away the fact
that a rain model can be excellent on the wet east coast and nearly useless in the
arid interior, so everything here is computed per station and reported with its
support attached.

``error_z`` is how many standard errors a station's error rate sits from the
overall rate under a binomial null.  It is support-aware, so a small station's
noise is not mistaken for bias; stations past |z| >= 2 are flagged.

Cramer's V accompanies the chi-square p-value on purpose: with tens of thousands
of rows a trivial difference is still "significant", and the effect size is what
says whether it matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

STATION_METRICS = {
    "f1": {"label": "F1", "lower_is_better": False},
    "precision": {"label": "Precision", "lower_is_better": False},
    "recall": {"label": "Recall", "lower_is_better": False},
    "accuracy": {"label": "Accuracy", "lower_is_better": False},
    "error_rate": {"label": "Error rate", "lower_is_better": True},
    "error_rate_delta": {"label": "Error rate vs overall", "lower_is_better": True},
    "error_z": {"label": "Error rate z-score", "lower_is_better": True},
    "positive_rate": {"label": "Observed rain rate", "lower_is_better": False},
    "support": {"label": "Validation days", "lower_is_better": False},
}

# metrics that are signed differences around zero, so they need a diverging scale
SIGNED_METRICS = {"error_rate_delta", "error_z"}

CALIBRATION_BINS = 10
SWEEP_STEPS = 99
FLAG_Z = 2.0
# a drawn curve needs far fewer points than a validation set has distinct scores
MAX_CURVE_POINTS = 2000


@dataclass
class Scored:
    model: str
    y_true: np.ndarray
    proba: np.ndarray
    threshold: float

    @property
    def y_pred(self) -> np.ndarray:
        return (self.proba >= self.threshold).astype(int)


def is_signed_metric(metric: str) -> bool:
    return metric in SIGNED_METRICS


def _safe(fn, *args, **kwargs) -> float:
    """Metrics are undefined on a degenerate split; report NaN rather than raising."""
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def global_metrics(scored: Scored) -> dict[str, float]:
    y, p, pred = scored.y_true, scored.proba, scored.y_pred
    return {
        "f1": _safe(f1_score, y, pred, zero_division=0),
        "precision": _safe(precision_score, y, pred, zero_division=0),
        "recall": _safe(recall_score, y, pred, zero_division=0),
        "accuracy": _safe(accuracy_score, y, pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y, pred),
        "roc_auc": _safe(roc_auc_score, y, p),
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else float("nan"),
        "support": float(len(y)),
    }


def _thin(frame: pd.DataFrame, limit: int = MAX_CURVE_POINTS) -> pd.DataFrame:
    """Subsample a curve for drawing, always keeping both endpoints.

    A PR curve has one point per distinct score - tens of thousands of them on a
    real validation set, far beyond both the renderer's row cap and the number of
    pixels available.  The reported AUCs are computed on the full arrays, so this
    only affects the drawn line.
    """
    if len(frame) <= limit:
        return frame
    keep = np.unique(
        np.concatenate([np.linspace(0, len(frame) - 1, limit).astype(int), [len(frame) - 1]])
    )
    return frame.iloc[keep].reset_index(drop=True)


def roc_points(scored: Scored) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(scored.y_true, scored.proba)
    return _thin(pd.DataFrame({"fpr": fpr, "tpr": tpr}))


def pr_points(scored: Scored) -> pd.DataFrame:
    precision, recall, _ = precision_recall_curve(scored.y_true, scored.proba)
    return _thin(pd.DataFrame({"recall": recall, "precision": precision}))


def calibration_bins(scored: Scored, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(scored.proba, edges[1:-1]), 0, bins - 1)
    frame = pd.DataFrame({"bin": index, "proba": scored.proba, "y": scored.y_true})
    grouped = frame.groupby("bin").agg(
        predicted=("proba", "mean"), observed=("y", "mean"), count=("y", "size")
    ).reset_index()
    return grouped


def threshold_sweep(scored: Scored, steps: int = SWEEP_STEPS) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, steps)
    rows = []
    for threshold in thresholds:
        pred = (scored.proba >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "f1": _safe(f1_score, scored.y_true, pred, zero_division=0),
            "precision": _safe(precision_score, scored.y_true, pred, zero_division=0),
            "recall": _safe(recall_score, scored.y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def confusion_frame(scored: Scored) -> pd.DataFrame:
    matrix = confusion_matrix(scored.y_true, scored.y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual: no rain", "Actual: rain"],
        columns=["Predicted: no rain", "Predicted: rain"],
    )


def per_station_metrics(
    scored: Scored,
    stations: pd.Series,
    min_support: int = 30,
) -> pd.DataFrame:
    """One row per station, with the support-aware error-rate z-score attached."""
    frame = pd.DataFrame({
        "station": pd.Series(stations).to_numpy(),
        "y": scored.y_true,
        "pred": scored.y_pred,
        "proba": scored.proba,
    }).dropna(subset=["station"])
    if frame.empty:
        return pd.DataFrame()

    frame["error"] = (frame["y"] != frame["pred"]).astype(int)
    overall_error = float(frame["error"].mean())

    rows = []
    for station, group in frame.groupby("station", sort=True):
        support = len(group)
        if support < min_support:
            continue
        y, pred = group["y"].to_numpy(), group["pred"].to_numpy()
        error_rate = float(group["error"].mean())
        # binomial standard error of this station's error rate under the null
        # that it shares the overall rate
        se = np.sqrt(overall_error * (1 - overall_error) / support)
        rows.append({
            "station": station,
            "support": support,
            "positive_rate": float(np.mean(y)),
            "predicted_positive_rate": float(np.mean(pred)),
            "error_rate": error_rate,
            "error_rate_delta": error_rate - overall_error,
            "error_z": float((error_rate - overall_error) / se) if se > 0 else float("nan"),
            "accuracy": _safe(accuracy_score, y, pred),
            "precision": _safe(precision_score, y, pred, zero_division=0),
            "recall": _safe(recall_score, y, pred, zero_division=0),
            "f1": _safe(f1_score, y, pred, zero_division=0),
            "roc_auc": _safe(roc_auc_score, y, group["proba"].to_numpy()),
        })
    return pd.DataFrame(rows)


def location_dependence_test(scored: Scored, stations: pd.Series) -> dict | None:
    """Chi-square test of independence between station and being wrong."""
    frame = pd.DataFrame({
        "station": pd.Series(stations).to_numpy(),
        "error": (scored.y_true != scored.y_pred).astype(int),
    }).dropna(subset=["station"])
    if frame.empty or frame["station"].nunique() < 2:
        return None

    table = pd.crosstab(frame["station"], frame["error"])
    if table.shape[1] < 2:
        return None

    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    smaller = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * smaller))) if n and smaller else float("nan")
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "n": n,
    }


def disparity_summary(frame: pd.DataFrame, metric: str) -> dict | None:
    if frame.empty or metric not in frame.columns:
        return None
    values = pd.to_numeric(frame[metric], errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "best": float(values.max()),
        "worst": float(values.min()),
        "spread": float(values.max() - values.min()),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
        "std": float(values.std(ddof=0)),
    }


def spatial_gradient(frame: pd.DataFrame, metric: str) -> dict | None:
    """Does the metric trend across the continent rather than scatter randomly?"""
    if frame.empty or metric not in frame.columns:
        return None
    subset = frame[[metric, "lat", "lon"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 3:
        return None

    out = {}
    for axis in ("lat", "lon"):
        out[f"{axis}_pearson"] = _safe(lambda a, b: stats.pearsonr(a, b)[0],
                                       subset[axis], subset[metric])
        out[f"{axis}_spearman"] = _safe(lambda a, b: stats.spearmanr(a, b)[0],
                                        subset[axis], subset[metric])
    return out
