# Note de refactoring — notebooks XGBoost

Adaptation des 9 notebooks de ce dossier pour qu'ils soient **reproductibles à partir du pipeline
de preprocessing du dépôt**, sur le modèle de `5.modelisation_linear_svm.ipynb` et
`5.modelisation_light_gbm.ipynb`, avec sérialisation `joblib` des modèles à chaque étape.

Date : 2026-08-11 · Branche : `XGBoost_appraoche`

---

## 1. Ce qui change en une phrase

Avant, chaque notebook refaisait son propre split, sa propre sélection de features et lisait le CSV
brut `weatherAUS.csv` (voire Google Drive). Maintenant tous consomment les **sorties du pipeline** :

| Notebook du pipeline | Sortie | Usage ici |
|---|---|---|
| `3.preprocessing_trend_group.ipynb` | `data/weatherAUS_trend_grouped.csv` | frame **pré-imputation** (NaN d'origine) → lags, features voisines, audit des manquants |
| `4_preprocessing_imputation.ipynb` | `data/weatherAUS_MICE_cyclic.csv` | index `(Date, Location)` + valeurs **non scalées** |
| `4_preprocessing_imputation.ipynb` | `data/weatherAUS_X_train.csv`, `weatherAUS_y_train.csv`, `weatherAUS_X_test.csv`, `weatherAUS_y_test.csv` | matrices de features et cible, déjà splitées / encodées / scalées |

Aucun notebook ne resplitte, ne réimpute, ni ne rescale quoi que ce soit.

---

## 2. L'invariant qui rend le recollement exact

C'est le point technique central du refactoring, et il a été **vérifié**, pas supposé :

1. `weatherAUS_MICE_cyclic.csv` est trié par `Date`, et `4_preprocessing_imputation.ipynb` y
   applique le même seuil `TEST_CUTOFF = 2016-01-01`. Donc ses **113 506 premières lignes _sont_**
   `weatherAUS_X_train.csv`, et ses **25 314 dernières _sont_** `weatherAUS_X_test.csv`
   (113 506 + 25 314 = 138 820).
   Vérifié colonne par colonne sur les colonnes non scalées (`Date_sin`, `Date_cos`, `Month`,
   `WindGustDir_sin`, `WindDir3pm_cos`), sur `RainToday`, et sur les deux fichiers `y`.
2. `(Date, Location)` est une clé **unique** et **identique** dans `weatherAUS_trend_grouped.csv`
   et `weatherAUS_MICE_cyclic.csv`, sur exactement le même jeu de 138 820 lignes.

Conséquences pratiques, présentes dans chaque notebook :

```python
meta = pd.read_csv(f'{DATA_DIR}/weatherAUS_MICE_cyclic.csv', usecols=[...], parse_dates=['Date'])
meta_train = meta.iloc[:len(X_train)].reset_index(drop=True)
meta_test  = meta.iloc[len(X_train):].reset_index(drop=True)
assert (meta_train['Date'] <  TEST_CUTOFF).all()
assert (meta_test['Date']  >= TEST_CUTOFF).all()
```

et, pour recoller un bloc de features construit sur `(Date, Location)` :

```python
def align_to_pipeline(frame, cols):
    key = pd.MultiIndex.from_frame(frame[['Date', 'Location']])
    assert key.is_unique
    out = frame.set_axis(key)[cols].reindex(pd.MultiIndex.from_frame(meta[['Date', 'Location']]))
    return out.reset_index(drop=True)
```

Le réalignement du masque d'imputation dans EXP-003 donne **0 ligne non appariée**.

**Effet de bord bienvenu :** le bug de split diagnostiqué en EXP-003 (masque trié par date appliqué
à des features restées dans l'ordre d'origine, groupées par `Location` → split
leave-stations-out accidentel) ne peut structurellement plus se produire : les lignes arrivent déjà
dans l'ordre chronologique du pipeline, et les `assert` ci-dessus le vérifient à chaque exécution.

---

## 3. Décisions prises (et pourquoi)

### 3.1 La drop-list est exprimée contre la matrice du pipeline

Le pipeline livre **29 colonnes**. On en retire 6 :

```python
DROP_COLS = [
    'WindGustDir', 'WindDir9am', 'WindDir3pm',  # labels encodés ; les sin/cos sont conservés
    'RainToday',                                # quasi-fuite
    'Humidity9am',                              # on garde Humidity3pm (plus prédictive)
    'Month',                                    # redondant avec Date_sin / Date_cos
]
```

Ce qui retombe **exactement** sur les compteurs déjà documentés dans les notebooks, donc les
expériences EXP-001 → 007 restent comparables entre elles :

| Bloc | Compte | Notebooks |
|---|---|---|
| base | **23** | EXP-001 |
| base + `Pressure_diff` + `Temp_diff` | **25** | EXP-002 → 007 |
| base + temporel (12) | **37** | EXP-004 |
| base + temporel + antécédent (7) | **44** | EXP-005 |
| base + temporel + spatial (7) | **44** | EXP-006, EXP-007, interpretability |
| base + indicatrices de manquant (7) | **32** | EXP-003 (`PLUS`) |

### 3.2 `Pressure_diff` / `Temp_diff` sont calculées sur les valeurs NON scalées

Elles viennent de `weatherAUS_MICE_cyclic.csv`, **pas** de `X_train`. Le `RobustScaler` du pipeline
centre et divise chaque colonne par son propre IQR ; `Pressure3pm_scaled - Pressure9am_scaled`
n'est donc pas proportionnel à la vraie différence de pression.

### 3.3 Aucun scaler ajouté

Le pipeline a déjà appliqué un `RobustScaler` ajusté sur le train uniquement, et les arbres de
gradient boosting sont de toute façon invariants à toute transformation monotone appliquée colonne
par colonne. Le commentaire d'origine des notebooks (« No `StandardScaler` is applied ») reste donc
valable, pour la même raison.

### 3.4 Les features retardées / voisines viennent du frame PRÉ-imputation

`weatherAUS_trend_grouped.csv` remplace l'ancien `weatherAUS.csv` brut. C'est strictement meilleur :
mêmes lignes que la matrice de modélisation (donc recollement exact au lieu d'un `merge` approximatif),
NaN d'origine conservés (donc aucune valeur reconstruite par MICE ne remonte dans un lag), et c'est
une sortie du pipeline plutôt qu'un fichier externe.

Adaptation à connaître : dans ce frame `RainToday` est un **booléen**, donc
`raw['RainToday'].map({'Yes': 1, 'No': 0})` devient `pre['RainToday'].astype(int)`.

### 3.5 Colab

Les `drive.mount()` et les chemins `MyDrive/Colab_Notebooks` sont supprimés. Le bloc
`pip install` conditionnel reste (inerte en local). Le chemin des données est :

```python
DATA_DIR = os.environ.get('WEATHER_DATA_DIR', '../../data')
```

→ reproductible par défaut en local, toujours redirigeable sur Colab via la variable d'environnement.

---

## 4. Sérialisation des modèles (`joblib`)

Chaque notebook définit un helper `save_model(...)`, calqué sur le `save_fig(...)` existant :

```python
MODEL_DIR = os.environ.get('WEATHER_MODEL_DIR', '../../models')

def save_model(obj, name, **meta):
    path = f'{MODEL_DIR}/{name}.joblib'
    joblib.dump({'model': obj, 'notebook': NOTEBOOK,
                 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 **meta}, path)
```

Les métadonnées portent selon les cas : `features` (ordre exact des colonnes attendues), `stage`,
`params` / `best_params`, `threshold`, `metrics`, `scale_pos_weight`, `split`, `K_neighbours`,
`neighbours`.

### Étapes sauvegardées

| Notebook | Artefacts |
|---|---|
| EXP-001 | `01_baseline`, `02_tuned`, `03/04/05_imbalance_*` (default / scale_pos_weight / SMOTE), `06_tuned_with_threshold`, `07_refit_85pct`, `08_calibrated_isotonic`, `09_calibrated_sigmoid` |
| EXP-002 | `01_regularized_tuned`, `02_early_stopped`, `03_refit_85pct`, `04_calibrated_isotonic`, `05_calibrated_sigmoid`, `06_tuned_with_threshold` |
| EXP-003 | `01_base`, `02_plus_indicators` |
| EXP-004 / 005 / 006 | un modèle par couple split × jeu de features (4, 6 et 6 respectivement) |
| EXP-007 sptune | `01_fresh_search`, `02_frozen__*`, `02_tuned__*` (2 splits chacun) |
| EXP-007 deploy | `deploy_01_refit_85pct`, `deploy_02_isotonic`, `deploy_03_production_full_train`, + le bundle `rain_xgb_k5_deploy.joblib` et `rain_xgb_k5_metadata.json` |
| interpretability | `interpretability_01_refit_K5` |

Une exécution complète de la série produit **44 artefacts** (9 + 6 + 2 + 4 + 6 + 6 + 5 + 5 + 1),
la plupart à ~3 Mo, soit de l'ordre de 100 Mo.

Tout part dans `models/` (le dossier prévu par le template cookiecutter). `.gitignore` a reçu
`models/*` + `!models/.gitkeep`, pour rester cohérent avec la règle `data/` déjà présente — sinon
ces 44 fichiers apparaîtraient en untracked.

---

## 5. Sorties des cellules effacées

**Tous les outputs stockés ont été vidés.** La source des données ayant changé, chaque nombre
mémorisé était périmé et aurait été trompeur à côté du nouveau code. Les fichiers passent de
~880 Ko à ~48 Ko.

Ce qui subsiste comme trace de l'exécution d'origine : la narration markdown des notebooks (les
chiffres de référence y sont cités), `TEMPORAL_result.txt` et `figures/`.

Les notebooks ont été repassés par `nbformat.normalize()` (ids de cellules) et valident tous.

---

## 6. Vérification — le refactoring reproduit-il les résultats ?

### EXP-004, comparé au `TEMPORAL_result.txt` versionné

| split / features | F1 (nouveau → ancien) | ROC-AUC | PR-AUC |
|---|---|---|---|
| chronological / base | 0.6565 → 0.6579 | 0.8820 → 0.8818 | 0.7307 → 0.7310 |
| chronological / base+temporal | 0.6657 → 0.6665 | 0.8861 → 0.8860 | 0.7401 → 0.7405 |
| leave-stations-out / base | 0.6584 → 0.6590 | 0.8773 → 0.8768 | 0.7283 → 0.7276 |
| leave-stations-out / base+temporal | 0.6663 → 0.6650 | 0.8824 → 0.8828 | 0.7386 → 0.7389 |

Écart max ~0.0014 sur toutes les métriques, comptes de features identiques (25 / 37), seuils
PR-crossover quasi identiques. Le verdict du notebook est conservé.

### Autres points de contrôle

- **EXP-003** : exécuté de bout en bout, reproduit son verdict d'origine (« GENUINE AMBIGUITY
  dominates », 0/3 signaux). Masque de manquants réaligné avec 0 ligne non appariée.
- **EXP-006** : exécuté de bout en bout, verdict conservé (« spatial neighbour context ADDS »).
  Sa ligne `chronological / base+temporal+spatial` donne F1 0.6828 / ROC 0.8958 / PR 0.7612, contre
  la référence 0.6807 / 0.8960 / 0.7608 que EXP-007 cite pour EXP-006. Ses lignes `base` et
  `base+temporal` sont **identiques au chiffre près** à celles de EXP-004, ce qui confirme aussi le
  déterminisme du chemin de données entre notebooks.
- **EXP-007 deploy** : exécuté de bout en bout — F1 0.6784 / ROC 0.8967 / PR 0.7613 contre
  0.686 / 0.898 / 0.763 documentés. Aller-retour de l'artefact `joblib` vérifié (F1 identique après
  rechargement).
- **EXP-007 sptune** : le balayage de graphes spatiaux retient K5 (PR-AUC 0.7614) ; K3 donne 0.7612
  contre 0.7608 documenté.
- **interpretability** : la section SHAP tourne et retrouve la structure attendue
  (`Humidity3pm` dominant, `nbr_Humidity3pm_t` 4ᵉ).

Le léger écart résiduel est attendu : les features de base passent désormais par l'imputation MICE
par groupe de tendance et le `RobustScaler` du pipeline, au lieu du chemin interne des notebooks.

### Couverture des tests

- Chemin d'installation (chargement → features → split → premier `fit`) : **exécuté pour les 9**.
- Exécution complète : EXP-003, EXP-004, EXP-006, EXP-007 deploy ; plus le balayage de
  EXP-007 sptune et la section SHAP de interpretability.
- **Non exécutés en entier** : EXP-001, EXP-002, EXP-005 — leurs parties coûteuses sont des
  `RandomizedSearchCV` (200 fits). Leurs boucles sont textuellement identiques à celles qui ont
  tourné, et un contrôle statique a vérifié que **tous** les appels `save_model` des 9 notebooks ne
  référencent que des variables déjà liées (et que toutes les cellules parsent).

---

## 7. Pour réexécuter la série

```bash
# 1. produire les données (une fois)
notebooks/3.preprocessing_trend_group.ipynb     ->  data/weatherAUS_trend_grouped.csv
notebooks/4_preprocessing_imputation.ipynb      ->  data/weatherAUS_MICE_cyclic.csv
                                                    data/weatherAUS_{X,y}_{train,test}.csv

# 2. puis, dans l'ordre, depuis notebooks/XGBoost_Sath/
EXP-001 -> EXP-002 -> EXP-003 -> EXP-004 -> EXP-005 -> EXP-006
        -> EXP-007_XGBoost_spatial_tuned -> EXP-007_XGBoost_deploy -> interpretability_XGB
```

Les notebooks sont indépendants les uns des autres (chacun reconstruit ce dont il a besoin depuis
`data/`) ; l'ordre ci-dessus n'est que l'ordre narratif des expériences.

Chemins surchargeables : `WEATHER_DATA_DIR` (défaut `../../data`) et `WEATHER_MODEL_DIR`
(défaut `../../models`).
