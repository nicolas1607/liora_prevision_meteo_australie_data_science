import streamlit as st

from tools import *

df_raw = load_data_raw()

st.title("👋 Introduction")

# aligner au centre
st.image(load_image("introduction"))

st.markdown("""

""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temps d'observation", "10 ans")
col2.metric("Nombre d'observations", df_raw.shape[0])
col3.metric("Variables", df_raw.shape[1])
col4.metric("Stations", df_raw['Location'].nunique())

st.subheader("Objectif du projet")
st.markdown("""
- **Classification binaire supervisée** : prédire la variable cible `RainTomorrow` à partir des variables météorologiques disponibles (0 = pas de pluie, 1 = pluie).

- **Contexte métier** : priorité à la classe « pluie » qui correspond aux erreurs les plus coûteuses (un jour de pluie manqué a des conséquences plus lourdes qu'une fausse alerte).
""")