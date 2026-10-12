# Rain-tomorrow model evaluation

A Streamlit app for validating any model saved under the project's `models/` on a
held-out set and breaking the result down **by weather station**, so you can see whether the model's
skill depends on where you are rather than having it averaged away into one F1 score.

It reads the project's shared `data/` and `models/` directories - it keeps no private
copy of either. Run it from the repository root:

```bash
pip install -r streamlit_diagnostic/requirements.txt   # its own pinned env, see below
streamlit run streamlit_diagnostic/streamlit_app.py
```

The CARTO basemaps on the map need a free API key (<https://carto.com/basemaps/apikey>);
without one CARTO serves an "API KEY REQUIRED" tile, so the app falls back to
OpenStreetMap. Put the key in `.streamlit/secrets.toml` at the repository root, which
is gitignored and must never be committed:

```toml
CARTO_API_KEY = "your-key"
```

or set the `CARTO_API_KEY` environment variable. `paths.carto_api_key()` reads it.

`WEATHER_DATA_DIR` and `WEATHER_MODEL_DIR` override the two locations, exactly as in
the `notebooks/XGBoost_Sath/` series. `paths.py` is the single place both are resolved;
nothing else in the app builds a data or model path. Every `.joblib` / `.pkl` file under
`models/` is offered, grouped by folder (`catboost/`, `light_gbm/`, `xgboost/` ...).

Its `requirements.txt` is deliberately separate from the root one: the app pins
`pandas==2.3.3` and needs `xgboost`, `lightgbm` and `catboost` to unpickle those model
families, while the root environment pins `pandas==3.0.2` and ships no xgboost or
catboost. Do not merge them.

## What it does

1. **Pick a validation set** — the raw `weatherAUS` history in the project's `data/`,
   the already-prepared test matrix, or your own upload (CSV / Parquet).
2. **Build a comparison of up to 8 models, from any folders** - pick a folder, tick
   model files and click _Add to comparison_; repeat in other folders to stack, say,
   LogisticRegression, XGBoost and CatBoost side by side. Nothing is loaded until you
   click, and each model can be removed with its ✕ (or _Remove all_). Files are listed
   by their real file name under the project's `models/` (prefixed with the folder only
   if two loaded files share a name), and each is scored at its own recorded decision
   threshold (`threshold`, or `seuil` in the LightGBM / LinearSVC bundles; 0.5 when
   none was saved).
3. **Read the metrics** — F1 / precision / recall / accuracy / balanced accuracy /
   ROC AUC, confusion matrices, ROC and PR curves, a calibration plot and a
   threshold sweep, all comparable across selected models.
4. **Read the map** — a folium map of Australia where each station is coloured by
   the metric you choose and sized by how many validation days it contributed.
5. **Read the diagnostics** — a chi-square test of whether error rate depends on
   station at all, per-station disparity statistics, a latitude/longitude gradient
   plot, and the list of stations whose error rate is statistically out of line.
6. **Predict one day** — the *Prediction* tab takes the original, unscaled observations
   for a station and date, typed by hand. Leave blank anything not observed; blanks
   stay missing, with no imputation. The app processes the input automatically, through
   the same `preprocessing.build_features` path the validation data takes:
   - RainToday is derived from Rainfall (> 1 mm);
   - the date and wind directions get their sin/cos encodings;
   - the base columns get the training RobustScaler;
   - the lag, 3-day-window and upwind-neighbour features are built from the year of
     history before that date.

   Every model in the comparison then returns its probability and Rain / No rain
   decision at its own threshold. A *Processed input* table shows every feature the
   models receive and where it came from. This tab works even without a validation source.
   - XGBoost, LightGBM, CatBoost, the decision trees, random forest and the dummy
     score through missing values. LogisticRegression and LinearSVC cannot; they
     report which entered fields to fill in.
   - The history ends on 2017-06-25. For a later date, the lag, window and neighbour
     features are missing, which only XGBoost uses, so its prediction leans on the
     same-day columns alone.

   Typing a historical day's values reproduces the batch features to 1e-14 and the
   same probabilities.

## Layout

```
streamlit_app.py             the UI and the flow between tabs
paths.py                     where data, models and resources live (the only path logic)
app/
  xgb_compat.py              loads the bundles despite the XGBoost pickle bug
  preprocessing.py           raw weatherAUS rows -> the models' 44 features
  data_io.py                 input detection, feature coverage, matrix assembly
  evaluation.py              global metrics, per-station metrics, bias statistics
  mapping.py                 the folium layers
  charts.py                  the Altair charts
  locations.py               station coordinates and name matching
  model_registry.py          model discovery by folder, loading, scoring adapter
  prediction.py              one raw station-day -> processed features -> predictions
  palette.py                 the chart palette
tools/
  fit_preprocessing.py       regenerates resources/preprocessing_stats.json
resources/                   app-owned, not shared with the project
  locations.csv              49 stations with coordinates
  preprocessing_stats.json   the recovered training transform
```

No module under `app/` builds a data or model path of its own - they all take them
from `paths.py` (`RAW_HISTORY`, `PREPARED_X`, `PREPARED_Y`, `PRE_IMPUTATION`,
`LOCATIONS_CSV`, `PREPROCESSING_STATS`, `discover_models()`).

## How each model family is scored

The notebooks saved their models in different shapes; `app/model_registry.py`
normalises them all. Every family except EXP-007 XGBoost expects the **23 base
columns** of the pipeline's scaled `X_train`, which both built-in validation sources
provide on that scale.

| Saved as | Scoring |
|---|---|
| bare estimator (`DummyClassifier`, `RandomForest`, `LGBMClassifier`, `Pipeline`) | features read from the estimator, `predict_proba` |
| dict bundle with `model` + `features` + `threshold` / `seuil` | as saved |
| bundle with `scaler` (logistic regression, LinearSVC baseline) | the bundle's own `StandardScaler` is applied first |
| bundle with `poly` + `cross_idx` (`logistic_regression_interactions`) | scaler, then the interaction terms are appended |
| `model` is a list (`logistic_regression_bootstrap_bagged`) | mean probability of the members |
| no `predict_proba` (LinearSVC) | sigmoid of `decision_function`; the threshold is mapped the same way, so every thresholded metric and ROC AUC are unchanged (the calibration plot is not meaningful) |
| EXP-007 XGBoost | the 44-feature path below; needs the raw history source |

Scored on `weatherAUS_X_test`, every bundle that recorded metrics reproduces them.
The one apparent exception is `logistic_regression_interactions`: its recorded F1 was
computed with `.predict()` (threshold 0.5), whereas the bundle stores 0.66, which is
the threshold the app applies. ROC AUC matches (0.8675).

`logistic_regression_bootstrap_bagged.joblib` carries a `coef_summary` DataFrame
pickled under pandas 3, which pandas 2.3.3 cannot rebuild. It is metadata only, so
the loader retries with pandas objects stubbed out and drops them.

The shared inputs are gitignored, so a fresh clone has none of them - the app checks at
startup and names the missing files. Copy the Google Drive `/_data` contents into
`data/`.

## Two things that had to be reverse-engineered

### The bundles do not load with `joblib.load`

Every booster in `models/` fails with `XGBoostError('input stream corrupted')`:
`Booster.__setstate__` hands the pickled blob to `XGBoosterUnserializeFromBuffer`,
which rejects it even though the blob is well-formed UBJSON written by the same
XGBoost version. The blob is a UBJSON object `{"Config": ..., "Model": ...}`, and
`"Model"` alone is exactly what `save_model(*.ubj)` writes — so `app/xgb_compat.py`
unpickles with the XGBoost classes stubbed out, slices `"Model"` out of the byte
buffer and feeds that to `Booster.load_model`. Only the training-time config is
dropped, which inference does not need.

### The bundles carry no preprocessing pipeline

They are bare boosters expecting 44 features, so the transform that produced their
training matrix was recovered from the two artefacts in the project's `data/`:

* rows of `weatherAUS_MICE_cyclic.csv` line up with `weatherAUS_X_test.csv` when the
  source is stable-sorted by `Date` alone and cut at 2016-01-01;
* with that pairing, a `RobustScaler` fitted on the pre-2016 rows reproduces the
  shipped matrix to **6e-14**, wind directions are label-encoded alphabetically, and
  `RainToday` is bool -> int.

The 21 engineered columns (lags, 3-day windows, diffs, upwind-neighbour aggregates)
are absent from `X_test`, so their definitions were pinned by scoring candidate
variants against the bundles' own recorded metrics. The decisive finding: **every
engineered feature is computed on raw, unscaled values**, while the 16 base weather
columns are scaled. Getting that backwards costs several points of AUC silently.

Measured on the 2016-01-01+ window, the reconstruction reproduces the two
chronologically-validated bundles closely:

| Bundle | ROC AUC | reported | F1 | reported |
|---|---|---|---|---|
| `EXP-007_02_frozen__chronological` | 0.8937 | 0.8966 | 0.6790 | 0.6823 |
| `EXP-007_02_tuned__chronological` | 0.8942 | 0.8964 | 0.6780 | 0.6843 |

The two leave-stations-out bundles score *higher* here (ROC AUC 0.934 and 0.941
against a reported 0.883), which is the expected direction: their reported figures
came from stations held out of training, whereas this window covers the stations
they saw.

One detail is worth more than the rest: the lag and neighbour terms are built from
`weatherAUS_trend_grouped.csv`, the **pre-imputation** frame (`note.md` 3.4), so no
value MICE invented can propagate into a lag. Building them from the imputed frame
instead costs about 0.002 ROC AUC and 0.003 F1.

Regenerate the constants with `python streamlit_diagnostic/tools/fit_preprocessing.py`.

## Input formats

**Raw weatherAUS** — `Date`, `Location` and the original observations (`data/weatherAUS_MICE_cyclic.csv`). The app
detects this (raw pressure is ~1000 hPa, scaled pressure is near 0), applies the
recovered transform and builds all 44 features itself. This is the only shape that
gives you the geographic views, because it is the only one carrying `Location`.

**Already-prepared** — a matrix that already holds the model features. The project's
`weatherAUS_X_test.csv` has the 23 base columns (enough for every family except
EXP-007 XGBoost) but only 23 of the 44 and no `Location`, so XGBoost cannot be scored
on it; the coverage tab says exactly which columns are absent. The missing
ones **cannot** be rebuilt from it, because the lag, window and neighbour terms live
on the raw scale and this matrix is already scaled.

Feed the widest history you have and use the date picker to choose the evaluation
window: features are built over everything first and sliced afterwards, so lag and
3-day terms at the start of the window still see the days before it.

## Reading the bias views

`error_z` is how many standard errors a station's error rate sits from the overall
rate, under a binomial null. It is support-aware, so a small station's noise is not
mistaken for bias. Stations past |z| ≥ 2 get a dashed ring on the map and a row in
the flagged table.

Cramér's V accompanies the chi-square p-value on purpose: with tens of thousands of
rows, a trivial difference is still "significant", and the effect size is what tells
you whether it matters.

Watch for the accuracy trap the map makes obvious. On the production bundle over
2016-01-01 → 2017-06-25, the arid interior (Woomera, Uluru, Mildura, Alice Springs)
has the *lowest* error rate and the *worst* F1 — it rarely rains there, so predicting
"no rain" is nearly free, while actual rain-detection skill is at its weakest. The wet
east coast (Norfolk Island, Coffs Harbour, Gold Coast, Wollongong) shows the opposite.
Error rate correlates with longitude at Spearman ≈ +0.50; F1 at ≈ −0.45. Pick the
metric that matches the question you are asking.
