import streamlit as st

from tools import *

df_raw = load_data_raw()

st.title("🔍 Exploration des données")

# col1, col2, col3 = st.columns(3)
# col1.metric("Observations", df_raw.shape[0])
# col3.metric("Variables", df_raw.shape[1])
# col2.metric("Stations", df_raw['Location'].nunique())

st.markdown(f"""
Le jeu de données `weatherAUS` compte **{df_raw.shape[1]}** variables 
et couvre **{df_raw['Location'].nunique()} stations météorologiques** australiennes
pour un total de **{df_raw.shape[0]} observations** journalières.
""")

st.subheader("Aperçu du jeu de données")
st.write(df_raw.head())

st.subheader(f"Valeurs manquantes : {df_raw.isna().sum().sum()} / {df_raw.size} ({(df_raw.isna().sum().sum() / df_raw.size * 100).round(1)}%)")
infos = pd.DataFrame({
    "Type": df_raw.dtypes.astype(str),
    "Non-nuls": df_raw.notna().sum(),
    "Taux de NaN (%)": (df_raw.isna().mean() * 100).round(1),
})
infos = infos.sort_values("Taux de NaN (%)", ascending=False)
st.dataframe(infos, use_container_width=True)

st.subheader(f"Présence de doublons : {df_raw.duplicated().sum()}")
