import streamlit as st

from tools import *

st.title("🤖 Modélisation")


tab_metrics, tab_models, tab_comparison, tab_selected_model = st.tabs([
    "Choix des métriques", 
    "Choix des modèles", 
    "Comparaison des modèles",
    "Modèle retenu",
])


with tab_metrics:
    
    st.subheader("Choix et hiérarchie des métriques")
    st.markdown("""
    L'accuracy est écartée étant donné qu'elle est peu informative sur des données déséquilibrées. 
    Le choix est donc porté sur des métriques plus pertinentes pour la classe minoritaire "Yes" :
    """)
    st.markdown("""
    <div class='title-center'><b>Indépendantes du seuil</b><br>(comparaison entre modèles)</div>
    """, unsafe_allow_html=True)
    _, col1, col2, _ = st.columns([1, 2, 2, 1])
    col1.markdown("<div class='markdown-metric'>Référence de comparaison<br><b>ROC-AUC</b></div>", unsafe_allow_html=True)
    col2.markdown("<div class='markdown-metric'>Indicateur complémentaire<br><b>PR-AUC</b></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='title-center'><b>Dépendantes du seuil</b><br>(évaluation opérationnelle d'un modèle)</div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.markdown("<div class='markdown-metric'>Métrique opérationnelle<br><b>F1-score 'Yes'</b></div>", unsafe_allow_html=True)
    col2.markdown("<div class='markdown-metric'>Indicateur complémentaire<br><b>Recall 'Yes'</b></div>", unsafe_allow_html=True)
    col3.markdown("<div class='markdown-metric'>Indicateur complémentaire<br><b>Précision 'Yes'</b></div>", unsafe_allow_html=True)
    st.markdown("""
    On choisit des métriques indépendantes du seuil pour comparer les modèles entre eux, et des métriques dépendantes du seuil pour évaluer la performance opérationnelle d'un modèle donné.

    L'asymétrie métier (un faux négatif coûte plus cher qu'un faux positif) est traitée ultérieurement par pondération des classes et ajustement du seuil.
    """)


with tab_models:
    
    st.subheader("Choix et hiérarchie des modèles")
    st.markdown("""
    **1. Baseline (établir un plancher de performance minimal)** : DummyClassifier

    **2. Linéaire : (tester l'hypothèse de séparabilité linéaire)** : LogisticRegression, LinearSVC

    **3. Bagging : (capturer les interactions non linéaires)** : DecisionTreeClassifier, RandomForestClassifier

    **4. Boosting (maximiser la performance prédictive sur la classe minoritaire)** : XGBoost, LightGBM, CatBoost
    """)

with tab_comparison:

    st.subheader("Comparaison des familles de modèles")

    LIBELLES = {
        "logistic_regression": "Linéaire — Logistic Regression",
        "linear_svc":          "Linéaire — Linear SVC",
        "decision_tree":       "Bagging — Decision Tree",
        "random_forest":       "Bagging — Random Forest",
        "light_gbm":           "Boosting — LightGBM",
        "catboost":            "Boosting — CatBoost",
        "xgboost":             "Boosting — XGBoost",
    }
    ORDRE = list(LIBELLES)

    familles = [
        d.name for d in MODEL_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.joblib"))
    ]
    familles.sort(key=lambda n: ORDRE.index(n) if n in ORDRE else len(ORDRE))

    if not familles:
        st.warning(f"Aucun modèle trouvé sous `{MODEL_DIR}`.")
        st.stop()

    col_famille, col_variante = st.columns(2)

    familles = [
        d.name for d in MODEL_DIR.iterdir()
        if d.is_dir() and lister_modeles(d)
    ]
    familles.sort(key=lambda n: ORDRE.index(n) if n in ORDRE else len(ORDRE))

    if not familles:
        st.warning(f"Aucun modèle (`.joblib` ou `.pkl`) sous `{MODEL_DIR}`.")
        st.stop()

    col_famille, col_variante = st.columns(2)

    famille = col_famille.selectbox(
        "Famille de modèle",
        options=familles,
        format_func=lambda n: LIBELLES.get(n, n.replace("_", " ").title()),
    )

    variantes = lister_modeles(MODEL_DIR / famille)

    variante = col_variante.selectbox(
        "Variante",
        options=list(variantes),
        format_func=lambda n: n.replace("_", " "),
    )

    bundle = load_model_path(str(variantes[variante]))

    if not isinstance(bundle, dict):
        st.warning(
            f"`{variante}` a été exporté en objet nu (`{type(bundle).__name__}`) "
            "et non au format bundle : ni métriques ni seuil disponibles."
        )
        bundle = {"model": bundle}

    # --- Scores -----------------------------------------------------
    metrics = bundle.get("metrics", {})
    seuil = bundle.get("threshold", bundle.get("seuil"))
    features = bundle.get("features", [])

    col1, col2 = st.columns(2)
    col1.metric("Seuil de décision", f"{seuil:.3f}" if isinstance(seuil, float) else "—")
    col2.metric("Nombre de features", len(features) if features else "—")

    if not metrics:
        st.info("Aucune métrique enregistrée dans ce bundle.")
    else:
        st.markdown("#### Scores")
        cles = list(metrics)
        cols = st.columns(min(len(cles), 4))
        for i, cle in enumerate(cles):
            v = metrics[cle]
            cols[i % len(cols)].metric(
                cle, f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
            )

    with st.expander("Hyperparamètres"):
        st.json(bundle["model"].get_params(), expanded=False)

    with st.expander("Liste des features"):
        st.write(features)

    # --- Figures ----------------------------------------------------
    dossier_figures = FIGURE_DIR / famille

    if not dossier_figures.is_dir():
        st.caption(f"Pas de dossier `reports/figures/{famille}/`.")
    else:
        figures = sorted(dossier_figures.glob("*.png"))
        if figures:
            st.markdown("#### Figures associées")
            onglets = st.tabs([f.stem.replace("_", " ").capitalize() for f in figures])
            for onglet, fig in zip(onglets, figures):
                with onglet:
                    st.image(load_figure(famille, fig.stem), width="stretch")
    

with tab_selected_model:
    
    st.subheader("Modèle retenu")
    st.markdown("""
    - **6 familles avec 23 features** pour prédire la variable cible (Logistic Regression, Linear SVC, Decision Tree, Random Forest, LightGBM, CatBoost)
    
    - **1 famille avec 44 features** pour prédire la variable cible (XGBoost)
    """)
    df_synthese_2 = load_csv("synthese_modeles_2")
    df_synthese_2 = df_synthese_2.style.map(lambda _: "background-color: #d4edda; color: #155724;", subset=pd.IndexSlice[:, ['AUC-ROC Test', 'PR-AUC Test', 'F1 "Yes" Test']])
    st.dataframe(df_synthese_2, width="stretch", height=800, use_container_width=True)
    st.markdown("""
    - **Le meilleur modèle tout périmètre confondu** : `xgboost_spatial_tuned` (AUC-ROC Test = 0.94, PR-AUC Test = 0.85, F1 "Yes" Test = 0.77)
    
    - **Le meilleur modèle à périmètre constant** : `catboost_final` (AUC-ROC Test = 0.88, PR-AUC Test = 0.72, F1 "Yes" Test = 0.65)
    
    - **Le concurrent immédiat à périmètre constant** : `light_gbm_threshold` (AUC-ROC Test = 0.88, PR-AUC Test = 0.72, F1 "Yes" Test = 0.61)
    """)
