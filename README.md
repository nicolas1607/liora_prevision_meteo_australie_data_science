# Prévision météo en Australie

## Pré-requis
```
Python 3.12.3
```

## Initialiser le projet
```
> git clone git@github.com:DataScientest-Studio/juin25_alt2_mle_meteo.git
> cd juin25_alt2_mle_meteo
> python3 -m venv env

> source env/bin/activate     # Pour Linux / macOS
> env\Scripts\activate        # Pour Windows (Command Prompt)
> env\Scripts\Activate.ps1    # Pour Windows (PowerShell)

> pip install -r requirements.txt

Ajouter les fichiers du dossier Google Drive /_data dans le dossier /data de votre projet
```

## Travailler sur un ticket
```
Les tickets sont disponibles dans l'onglet "Projects" du repo Github

> git checkout main
> git pull origin main (pour être sûr d'être à jour)
> git checkout -b ma_branche (ex : modelisation-linear-regression)
> git commit -m "implémentation du modèle de régression linéaire"
> git push origin modelisation-linear-regression

Créer ensuite une pull request dans l'onglet "Pull requests" du repo Github
```

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
