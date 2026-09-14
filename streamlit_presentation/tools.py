import pickle
import joblib
import pandas as pd
import streamlit as st

from PIL import Image

from paths import *


def load_data_raw():
    return pd.read_csv(DATA_DIR / "weatherAUS.csv")

def load_data_trend_group():
    return pd.read_csv(DATA_DIR / "weatherAUS_trend_grouped.csv")

def load_data_imputed():
    # return X_test, y_test, X_train, y_train
    X_test = pd.read_csv(DATA_DIR / "weatherAUS_X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "weatherAUS_y_test.csv")
    X_train = pd.read_csv(DATA_DIR / "weatherAUS_X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "weatherAUS_y_train.csv")
    return X_test, y_test, X_train, y_train

def load_image(filename):
    return Image.open(f"{FIGURE_DIR}/{filename}.png")

def load_visualisation(filename):
    return Image.open(f"{VIZ_DIR}/{filename}.png")

def load_figure(model_name, filename):
    return Image.open(FIGURE_DIR / f"{model_name}/{filename}.png")

def lister_modeles(dossier):
    """Retourne {nom_relatif: chemin} pour tous les modèles sous `dossier`."""
    fichiers = [
        f for f in sorted(dossier.rglob("*"))
        if f.is_file() and f.suffix in {".joblib", ".pkl", ".pickle"}
    ]
    return {
        f.relative_to(dossier).with_suffix("").as_posix(): f
        for f in fichiers
    }

@st.cache_resource
def load_model_path(chemin):
    chemin = Path(chemin)
    try:
        return joblib.load(chemin)
    except Exception:
        with open(chemin, "rb") as f:
            return pickle.load(f)

def load_csv(filename):
    return pd.read_csv(REPORT_DIR / f"{filename}.csv")

def load_html(filename):
    with open(VIZ_DIR / f"{filename}.html", "r") as f:
        html = f.read()
    return html