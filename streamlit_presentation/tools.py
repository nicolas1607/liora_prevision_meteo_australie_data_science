import joblib
import pandas as pd

from PIL import Image

from paths import *


def load_data_raw():
    df = pd.read_csv(DATA_DIR / "weatherAUS.csv")
    print(len(df))
    return df

def load_data_trend_group():
    df = pd.read_csv(DATA_DIR / "weatherAUS_trend_grouped.csv")
    print(len(df))
    return df

def load_data_imputed():
    # return X_test, y_test, X_train, y_train
    X_test = pd.read_csv(DATA_DIR / "weatherAUS_X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "weatherAUS_y_test.csv")
    X_train = pd.read_csv(DATA_DIR / "weatherAUS_X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "weatherAUS_y_train.csv")
    print("X_test :", len(X_test), "y_test :", len(y_test), "X_train :", len(X_train), "y_train :", len(y_train))
    return X_test, y_test, X_train, y_train

def load_image(filename):
    img = Image.open(f"{FIGURE_DIR}/{filename}.png")
    return img

def load_visualisation(filename):
    img = Image.open(f"{VIZ_DIR}/{filename}.png")
    return img

def load_figure(model_name, filename):
    fig = Image.open(FIGURE_DIR / f"{model_name}/{filename}.png")
    return fig

def load_model(filename):
    model = joblib.load(MODEL_DIR / f"{filename}.joblib")
    return model

def load_csv(filename):
    df = pd.read_csv(REPORT_DIR / f"{filename}.csv")
    return df

def load_html(filename):
    with open(VIZ_DIR / f"{filename}.html", "r") as f:
        html = f.read()
    return html