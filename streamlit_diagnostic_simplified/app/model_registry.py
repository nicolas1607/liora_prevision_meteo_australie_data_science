"""Discover, load and score every model saved under the shared ``models/`` directory.

The model families were saved by different notebooks in different shapes, so
loading normalises them all into one ``LoadedModel``:

* a **bare estimator** (``DummyClassifier``, ``RandomForestClassifier``,
  ``LGBMClassifier``, a ``Pipeline`` ...) - features come from the estimator;
* a **bundle dict** with the estimator under ``model`` plus metadata -
  ``features``, ``threshold`` (or ``seuil`` in the LightGBM / LinearSVC
  notebooks), ``metrics``, ``params`` ...  Two keys change how it is scored:

  - ``scaler`` - a ``StandardScaler`` fitted on the pipeline's (already
    RobustScaler-scaled) ``X_train``; the estimator expects its output;
  - ``poly`` + ``top_k`` + ``cross_idx`` + ``cross_names`` - the logistic
    regression interaction terms, appended after scaling.

* the bagged logistic regression stores a **list** of estimators whose
  probabilities are averaged.

Estimators without ``predict_proba`` (``LinearSVC``) are scored through
``decision_function`` mapped by a sigmoid.  The map is monotone, so ROC AUC, F1
and every thresholded metric are unchanged; the recorded threshold is mapped the
same way.  Only the calibration plot is not meaningful for those models.

All families except the EXP-007 XGBoost bundles expect the 23 base columns, which
both validation sources provide on the training scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import paths
from xgb_compat import is_stub, load_bundle, load_without_pandas

# metrics this app no longer reports; hidden from the recorded figures too
HIDDEN_METRICS = {"pr_auc", "mcc", "brier", "fpr", "fnr"}
# bundle keys that are scoring machinery or too bulky to list as a field
_INTERNAL_KEYS = {
    "model", "features", "scaler", "poly", "top_k", "cross_idx", "cross_names",
    "metrics", "params", "best_params", "coef_summary", "base_estimator", "_recovered",
}


@dataclass(frozen=True)
class ModelEntry:
    key: str
    label: str
    path: Path
    folder: str = ""


@dataclass
class LoadedModel:
    entry: ModelEntry
    estimator: Any
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------ description

    @property
    def score_kind(self) -> str:
        if isinstance(self.estimator, (list, tuple)):
            return f"mean probability of {len(self.estimator)} models"
        if hasattr(self.estimator, "predict_proba") or _is_booster(self.estimator):
            return "probability"
        if hasattr(self.estimator, "decision_function"):
            return "sigmoid(decision function)"
        return "hard label"

    @property
    def features(self) -> list[str]:
        """The columns the estimator itself receives, in order."""
        listed = self.meta.get("features")
        if listed is not None and len(listed):
            return [str(f) for f in listed]
        return _estimator_features(self._first_estimator)

    @property
    def input_features(self) -> list[str]:
        """The columns the validation data must supply.

        Equal to ``features`` except for the interaction bundle, whose cross terms
        are built here from the base columns.
        """
        if self.meta.get("poly") is None:
            return self.features
        scaler = self.meta.get("scaler")
        names = getattr(scaler, "feature_names_in_", None)
        if names is not None:
            return [str(n) for n in names]
        cross = set(self.meta.get("cross_names") or [])
        return [f for f in self.features if f not in cross]

    @property
    def threshold(self) -> float:
        """Decision threshold on the score this model reports (always in [0, 1])."""
        value = self.meta.get("threshold", self.meta.get("seuil"))
        if self.score_kind.startswith("sigmoid"):
            return float(_sigmoid(float(value) if value is not None else 0.0))
        return float(value) if value is not None else 0.5

    @property
    def _first_estimator(self):
        if isinstance(self.estimator, (list, tuple)):
            return self.estimator[0] if self.estimator else None
        return self.estimator

    # ---------------------------------------------------------------- scoring

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the bundle's own scaler and interaction terms, if it has any."""
        X = frame[self.input_features]
        scaler = self.meta.get("scaler")
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        poly = self.meta.get("poly")
        if poly is not None:
            crossed = poly.transform(X[self.meta["top_k"]])[:, self.meta["cross_idx"]]
            X = pd.concat(
                [X, pd.DataFrame(crossed, columns=self.meta["cross_names"], index=X.index)],
                axis=1,
            )
        return X[self.features]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Positive-class score in [0, 1], as a 1-D array."""
        X = self._transform(frame)
        if isinstance(self.estimator, (list, tuple)):
            return np.mean([_score(m, X) for m in self.estimator], axis=0)
        return _score(self.estimator, X)


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _is_booster(estimator) -> bool:
    return type(estimator).__name__ == "Booster" and type(estimator).__module__.startswith("xgboost")


def _score(estimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        return proba[:, 1] if proba.ndim == 2 else proba
    if _is_booster(estimator):
        # a bundle recovered by xgb_compat is a bare Booster; for binary:logistic
        # its raw prediction already *is* the positive-class probability
        import xgboost as xgb

        return np.asarray(estimator.predict(xgb.DMatrix(X.to_numpy(), feature_names=list(X.columns))))
    if hasattr(estimator, "decision_function"):
        return _sigmoid(estimator.decision_function(X))
    return np.asarray(estimator.predict(X), dtype=float)


def _estimator_features(estimator) -> list[str]:
    for attribute in ("feature_names_in_", "feature_name_", "feature_names_", "feature_names"):
        names = getattr(estimator, attribute, None)
        if names is not None and len(names):
            return [str(n) for n in names]
    return []


# ------------------------------------------------------------------ discovery


def discover() -> dict[str, list[ModelEntry]]:
    """Every model file in the shared models directory, grouped by folder."""
    return {
        folder: [
            ModelEntry(key=f"{folder}/{path.name}", label=path.name, path=path, folder=folder)
            for path in files
        ]
        for folder, files in paths.discover_models().items()
    }


def load(entry: ModelEntry) -> LoadedModel:
    try:
        bundle = load_bundle(entry.path)
    except (NotImplementedError, TypeError, AttributeError) as original:
        # a side DataFrame pickled under another pandas major version
        bundle = load_without_pandas(entry.path)
        if not isinstance(bundle, dict) or is_stub(bundle.get("model")):
            raise original
    if not isinstance(bundle, dict) or "model" not in bundle:
        bundle = {"model": bundle}
    estimator = bundle["model"]
    meta = {k: v for k, v in bundle.items() if k != "model" and not is_stub(v)}
    return LoadedModel(entry=entry, estimator=estimator, meta=meta)


def can_score(model: LoadedModel) -> bool:
    """A calibration map or a non-estimator cannot score feature rows."""
    estimator = model._first_estimator
    if estimator is None or not model.features:
        return False
    return any(
        hasattr(estimator, attr) for attr in ("predict_proba", "decision_function", "predict")
    ) and type(estimator).__name__ not in {"IsotonicRegression", "_SigmoidCalibration"}


# ---------------------------------------------------------------- description


def _estimator_name(model: LoadedModel) -> str:
    estimator = model.estimator
    if isinstance(estimator, (list, tuple)):
        inner = type(estimator[0]).__name__ if estimator else "?"
        return f"{len(estimator)} x {inner}"
    if hasattr(estimator, "steps"):
        return "Pipeline(" + " -> ".join(type(step).__name__ for _, step in estimator.steps) + ")"
    return type(estimator).__name__


def describe(model: LoadedModel) -> dict[str, str]:
    meta = model.meta
    raw_threshold = meta.get("threshold", meta.get("seuil"))
    out = {
        "File": model.entry.path.name,
        "Folder": model.entry.folder,
        "Estimator": _estimator_name(model),
        "Score": model.score_kind,
        "Features": str(len(model.features)),
        "Threshold": f"{model.threshold:.2f}"
        + (f" (recorded {raw_threshold})" if model.score_kind.startswith("sigmoid") else "")
        + ("" if raw_threshold is not None else " (default)"),
    }
    if meta.get("scaler") is not None:
        out["Own scaler"] = type(meta["scaler"]).__name__
    if meta.get("poly") is not None:
        out["Interaction terms"] = str(len(meta.get("cross_names") or []))
    for key, value in meta.items():
        if key in _INTERNAL_KEYS or key in {"threshold", "seuil"}:
            continue
        if isinstance(value, (str, int, float, np.number, bool)):
            out[key.replace("_", " ").capitalize()] = (
                f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            )
    if meta.get("_recovered"):
        out["Loaded via"] = "UBJSON recovery (xgb_compat)"
    return out


def hyper_parameters(model: LoadedModel) -> dict | None:
    params = model.meta.get("params") or model.meta.get("best_params")
    if isinstance(params, dict):
        return params
    estimator = model._first_estimator
    if hasattr(estimator, "get_params") and not hasattr(estimator, "steps"):
        try:
            return estimator.get_params()
        except Exception:
            return None
    return None


def reported_metrics(model: LoadedModel) -> dict[str, str]:
    """The figures the training notebook recorded, if the bundle carries any."""
    metrics = model.meta.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    out = {}
    for key, value in metrics.items():
        if str(key).lower() in HIDDEN_METRICS or isinstance(value, (dict, list)):
            continue
        out[key] = f"{value:.4f}" if isinstance(value, (int, float, np.number)) else str(value)
    return out
