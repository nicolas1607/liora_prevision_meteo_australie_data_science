import streamlit as st

from tools import *

st.title("📊 Visualisation")

st.markdown("""

""")


tab_correlation, tab_distribution, tab_precipitation, tab_distribution_spatial, tab_spatio_temp = st.tabs([
    "Corrélation entre variables", 
    "Distribution des variables", 
    "Disparités de précipitation", 
    "Distribution spatiale",
    "Analyse spatio-temporelle",
])


with tab_correlation:
    
    st.subheader("Corrélation entre les variables : présence de multicolinéarité")
    st.markdown("""
    L’application d’un **clustering hiérarchique** à la **matrice de corrélation** permet d’identifier des familles de variables au comportement très similaire.
    """)
    st.image(load_visualisation("matrice_correlation_clustering"))
    st.markdown("""
    On remarque alors plusieurs familles de variables fortement corrélées entre elles :
    - **La pression atmosphérique** : Pressure(9am/3pm).
    - **Les températures** : MaxTemp, MinTemp, Temp(9am/3pm).
    - **La dynamique des vents** : WindGustSpeed, WindSpeed(9am/3pm).
    - **L’hygrométrie et précipitations** : Sunshine, Cloud(9am/3pm), Humidity(9am/3pm).
    - **Les oppositions physiques** : les corrélations négatives soulignent des réalités logiques, comme Sunshine qui est fortement anti-corrélé avec la présence de nuages et l'humidité.
    """)


with tab_distribution:
    
    st.subheader("Distribution de la variable cible : déséquilibre de classe")
    st.markdown("""
    La variable cible `RainTomorrow` est **très déséquilibrée** : environ **78 % / 22 %** (pas de pluie / pluie). 
    """)
    st.image(load_visualisation("distribution_variable_cible"))
    st.markdown("""
    Ce déséquilibre conditionne toute la suite : choix des métriques, `class_weight='balanced'` et ajustement du seuil de décision.
    """)
    
    st.subheader("Distribution des variables numériques")
    st.image(load_visualisation("distribution_variables_numeriques"))
    

with tab_precipitation:
    
    st.subheader("Disparités de précipitations entre les villes")
    # TODO
    
    
with tab_distribution_spatial:
    
    st.subheader("Distribution spatiale et analyse géographique")
    
    with st.expander("Pourcentage de valeurs nulles par Location"):
        st.markdown("""        
        Les valeurs manquantes semblent concentrées dans certaines localisations, principalement côtières, mais pas uniquement. 
        
        Cette distribution suggère que les données manquantes dépendent de la qualité de collecte des stations plutôt que d’un simple effet géographique.
        """)
        st.subheader("Evaporation : pourcentage de valeurs nulles par Location")
        st.components.v1.html(load_html("location_evaporation_nan_map"), height=800, scrolling=True)    
        st.subheader("Sunshine : pourcentage de valeurs nulles par Location")
        st.components.v1.html(load_html("location_sunshine_nan_map"), height=800, scrolling=True)

    with st.expander("Nombre de variables entièrement nulles par Location"):
        st.markdown("""
        Certaines localisations présentent un nombre important de variables entièrement nulles.
        """)
        st.subheader("Nombre de variables entièrement nulles par Location")
        st.components.v1.html(load_html("location_fully_missing_variables_map"), height=800, scrolling=True)
        st.markdown("""
        - **Localisations avec 4 variables totalement nulles**

            - BadgerysCreek : Evaporation, Sunshine, Cloud9am, Cloud3pm
            - GoldCoast : Evaporation, Sunshine, Cloud9am, Cloud3pm
            - Nhil : Evaporation, Sunshine, Cloud9am, Cloud3pm
            - NorahHead : Evaporation, Sunshine, Cloud9am, Cloud3pm
            - Tuggeranong : Evaporation, Sunshine, Cloud9am, Cloud3pm
            - Walpole : Evaporation, Sunshine, Cloud9am, Cloud3pm
            - Witchcliffe : Evaporation, Sunshine, Cloud9am, Cloud3pm
        """)
        st.markdown("""
        - **Localisations avec 6 variables totalement nulles**

            - MountGinini : Evaporation, Sunshine, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm
            - Newcastle : Evaporation, Sunshine, WindGustDir, WindGustSpeed, Pressure9am, Pressure3pm
            - Penrith : Evaporation, Sunshine, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm
            - SalmonGums : Evaporation, Sunshine, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm
        """)
        st.markdown("""
        Toutes les villes concernées, mis à part Newcastle, présentent les mêmes variables entièrement manquantes : `Evaporation`, `Sunshine`, `Cloud9am` et `Cloud3pm`, 
        
        Ces représentations confirment un **schéma de collecte incomplet** propre à certaines stations **plutôt qu’une absence aléatoire**.
        """)
    
with tab_spatio_temp:
    
    st.subheader("Analyse spatio-temporelle")
    st.image(load_visualisation("presence_pluie_ville_source"))
    st.image(load_visualisation("effet_spatial_decalage_temporel"))
