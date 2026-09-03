import streamlit as st
import pandas as pd

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
    - **Indépendantes du seuil (comparaison entre modèles)**
        - ROC-AUC (référence de comparaison) : mesure la capacité à distinguer les deux classes indépendamment du seuil de décision, tout en restant robuste au déséquilibre.
        - PR-AUC (indicateur complémentaire) : la ROC-AUC peut paraître élevée même sur des données déséquilibrées, et la PR-AUC reflète les performances sur la classe minoritaire.
    """)
    st.markdown("""
    - **Dépendantes du seuil (évaluation opérationnelle d'un modèle)**
        - F1-score "Yes" (métrique opérationnelle) : combine précision et recall sur la classe pluie et évalue la performance sur la classe minoritaire plutôt que de la diluer dans une moyenne.
        - Recall "Yes" : un recall faible signale trop de jours de pluie manqués (faux négatifs).
        - Précision "Yes" : une précision faible signale trop de fausses alertes (faux positifs).
    """)
    st.markdown("""
    L'accuracy est écartée : elle est peu informative sur des données déséquilibrées.

    On choisit des métriques indépendantes du seuil pour comparer les modèles entre eux, et des métriques dépendantes du seuil pour évaluer la performance opérationnelle d'un modèle donné.

    L'asymétrie métier (un faux négatif coûte plus cher qu'un faux positif) est traitée ultérieurement par pondération des classes et ajustement du seuil.
    """)


with tab_models:
    
    st.subheader("Choix et hiérarchie des modèles")
    st.markdown("""
    **1. Baseline : établir un plancher de performance minimal**
    - Dummy Classifier (référence minimale indispensable) : avec 78 % de jours sans pluie, un modèle prédisant systématiquement la classe majoritaire atteint déjà une accuracy élevée, ce qui justifie l'abandon de cette métrique au profit du F1-score et de le ROC-AUC.

    **2. Linéaire : tester l'hypothèse de séparabilité linéaire**
    - Logistic Regression : apporte un modèle probabiliste directement interprétable via ses coefficients, et sert de référence pour juger de l'apport des approches non linéaires.
    - Linear SVC : complète cette étape en optimisant une marge de séparation plutôt qu'une vraisemblance ; sa comparaison avec la régression logistique permet d'évaluer la sensibilité du problème à la fonction de perte choisie.

    **3. Bagging : capturer les interactions non linéaires**
    - Decision Tree : modélise nativement les effets de seuil que les modèles linéaires ne peuvent représenter sans termes d'interaction explicites.
    - Random Forest : corrige la variance élevée et la tendance au surapprentissage de l'arbre isolé par agrégation d'arbres décorrélés, tout en restant robuste aux valeurs extrêmes.

    **4. Boosting : maximiser la performance prédictive sur la classe minoritaire**
    - Le boosting construit les arbres séquentiellement en pondérant les erreurs résiduelles, stratégie particulièrement adaptée à un jeu déséquilibré où les jours de pluie sont précisément les observations les plus difficiles à classer.
    - XGBoost, LightGBM et CatBoost : les trois implémentations retenues reposent sur le même principe mais diffèrent par la stratégie de croissance des arbres et de régularisation. Leur comparaison permet de vérifier la stabilité des performances au sein de cette famille.
    - Ces modèles exposent des mesures d'importance des variables (feature importance) qui restituent une part de l'interprétabilité perdue par rapport aux modèles linéaires.
    """)

with tab_comparison:
    
    st.subheader("Comparaison des familles de modèles")
    st.info("TODO : charger le tableau de synthèse produit par `6_synthese_modeles.ipynb`.")

    # Exemple de structure attendue :
    # df = pd.read_csv("resources/synthese_modeles.csv")
    # st.dataframe(df, use_container_width=True)


with tab_selected_model:
    
    st.subheader("Modèle retenu")
    st.markdown("""
    La variante XGBoost retenue s'appuie sur **44 variables** (décalages temporels, moyennes glissantes, variables de voisinage spatial) contre **23** pour les autres familles.
    
    Le gain provient donc de l'enrichissement des variables, non d'une supériorité algorithmique.
    """)
