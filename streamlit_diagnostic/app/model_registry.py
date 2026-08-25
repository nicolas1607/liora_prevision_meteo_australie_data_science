"""Discover, load and describe the EXP-007 bundles.

A bundle is the dict the notebooks' ``save_model`` helper writes: the fitted
estimator under ``model`` plus metadata - ``features`` (the exact column order the
booster expects), ``threshold``, ``metrics``, ``params``, ``stage``, ``split``.

``models/`` is shared with the whole EXP-001..007 series, so discovery is limited
to ``paths.MODEL_GLOB``; the earlier experiments use different feature blocks and
this app cannot score them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import paths
from xgb_compat import load_bundle

METRIC_KEYS = ["F1", "precision", "recall", "ROC_AUC", "PR_AUC", "threshold"]


@dataclass(frozen=True)
class ModelEntry:
    key: str
    label: str
    path: Path


@dataclass
class LoadedModel:
    entry: ModelEntry
    estimator: Any
    meta: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return list(self.meta.get("features") or [])

    @property
    def threshold(self) -> float:
        value = self.meta.get("threshold")
        return float(value) if value is not None else 0.5

    @property
    def isotonic(self):
        """The calibration map, for a bundle that carries one."""
        return self.estimator

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        """Positive-class probability, as a 1-D array.

        A bundle recovered by ``xgb_compat`` is a bare ``Booster`` rather than the
        ``XGBClassifier`` wrapper, so it has no ``predict_proba`` and needs its
        feature names attached to the DMatrix.  For ``binary:logistic`` the raw
        booster prediction already *is* the positive-class probability.
        """
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(matrix)
            return np.asarray(proba)[:, 1] if np.ndim(proba) == 2 else np.asarray(proba)

        import xgboost as xgb

        names = self.features or None
        dmatrix = xgb.DMatrix(matrix, feature_names=names)
        return np.asarray(self.estimator.predict(dmatrix))


def _label_for(path: Path) -> str:
    """Turn EXP-007_02_tuned__chronological into 'Tuned - chronological'."""
    stem = re.sub(r"^EXP-007[_-]*", "", path.stem)
    stem = re.sub(r"^deploy[_-]*", "deploy ", stem)
    stem = re.sub(r"^\d+[_-]*", "", stem)
    text = stem.replace("__", " - ").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else path.stem


def discover() -> list[ModelEntry]:
    """Every EXP-007 bundle in the shared models directory, production first."""
    entries = [
        ModelEntry(key=path.stem, label=_label_for(path), path=path)
        for path in paths.discover_models()
    ]
    return sorted(entries, key=lambda e: ("production" not in e.key, e.key))


def load(entry: ModelEntry) -> LoadedModel:
    bundle = load_bundle(entry.path)
    estimator = bundle.get("model")
    meta = {k: v for k, v in bundle.items() if k != "model"}
    return LoadedModel(entry=entry, estimator=estimator, meta=meta)


def is_calibrator(model: LoadedModel) -> bool:
    """A calibration map maps scores to probabilities; it cannot score features."""
    name = type(model.estimator).__name__
    if any(kind in name for kind in ("Isotonic", "Sigmoid", "Calibrat")):
        return True
    if hasattr(model.estimator, "predict_proba"):
        return False
    # a bundle recovered by xgb_compat is a bare Booster: a scorer that simply
    # lost its sklearn wrapper, not a calibration map
    return name != "Booster"


def describe(model: LoadedModel) -> dict[str, str]:
    meta = model.meta
    out = {
        "Estimator": type(model.estimator).__name__,
        "Notebook": str(meta.get("notebook", "-")),
        "Stage": str(meta.get("stage", "-")),
        "Features": str(len(model.features)),
        "Threshold": f"{model.threshold:.2f}",
    }
    for key in ("split", "spatial_graph", "scale_pos_weight", "calibration_rows", "cv_f1"):
        if key in meta:
            value = meta[key]
            out[key.replace("_", " ").title()] = (
                f"{value:.4f}" if isinstance(value, float) else str(value)
            )
    out["Created"] = str(meta.get("created_utc", "-"))
    if meta.get("_recovered"):
        out["Loaded via"] = "UBJSON recovery (xgb_compat)"
    return out


def reported_metrics(model: LoadedModel) -> dict[str, str]:
    """The figures the training notebook recorded, if the bundle carries any."""
    metrics = model.meta.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    out = {}
    for key in METRIC_KEYS:
        if key in metrics:
            value = metrics[key]
            out[key] = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
    for key, value in metrics.items():
        if key not in out and not isinstance(value, (dict, list)):
            out[key] = str(value)
    return out
