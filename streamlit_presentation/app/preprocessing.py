import streamlit as st

# from paths import DATA_DIR   # à activer selon le contenu de ton paths.py

st.title("🧹 Preprocessing")

st.markdown("""
L’objectif de cette étape est de **préparer les données** pour l’entraînement des modèles.
""")


tab_transformation, tab_clustering, tab_imputation, tab_etude_vent, tab_recap = st.tabs([
    "Transformation des données", 
    "Clustering exploratoire", 
    "Imputation des valeurs manquantes", 
    "Étude de la direction du vent",
    "Récapitulatif des actions",
])


with tab_transformation:
    
    st.subheader("Transformation des données")
    # TODO
    
    
with tab_clustering:
    
    st.subheader("Clustering exploratoire")
    # TODO
    
    
with tab_imputation:
    
    st.subheader("Imputation des valeurs manquantes")
    # TODO
    
    
with tab_etude_vent:
    
    st.subheader("Étude de la direction du vent")
    # TODO
    
    
with tab_recap:
    
    st.subheader("Récapitulatif des actions")
    # TODO
    