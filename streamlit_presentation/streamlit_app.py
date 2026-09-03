import streamlit as st

st.set_page_config(
    page_title="Prévision météo en Australie",
    page_icon="🌧️",
    layout="wide",
)

with st.sidebar:
    st.title("Prévision météo en Australie 🌧️")

pages = [
    st.Page("app/introduction.py",  title="Introduction",  icon="👋", default=True),
    st.Page("app/exploration.py",   title="Exploration",   icon="🔍"),
    st.Page("app/visualisation.py", title="Visualisation", icon="📊"),
    st.Page("app/preprocessing.py", title="Preprocessing", icon="🧹"),
    st.Page("app/modelisation.py",  title="Modélisation",  icon="🤖"),
    st.Page("app/conclusion.py",    title="Conclusion",    icon="🎯"),
]

nav = st.navigation(pages)
nav.run()
