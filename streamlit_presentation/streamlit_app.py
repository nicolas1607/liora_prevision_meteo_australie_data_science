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


# STYLE CSS

st.markdown("""
<style>

.title-center {
    text-align: center;
    font-size: 16px;
    margin-top: 1em;
    margin-bottom: 1em;
    b{
        font-size: 24px;
    }
}

.markdown-metric{
    background: linear-gradient(135deg, #f8f9fc 0%, #eef1f8 100%);
    border: 1px solid #e0e4ed;
    border-radius: 10px;
    padding: 18px 12px;
    margin: 0 auto 3em auto;
    width: 60%;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.markdown-metric:hover{
    transform: translateY(-4px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

.markdown-metric b{
    font-size: 24px;
    display: block;
    margin-top: 6px;
    color: #1f2d47;
}

</style>
""", unsafe_allow_html=True)