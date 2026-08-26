"""
Exécute le pipeline de notebooks du projet, dans l'ordre,
ou un seul notebook si on le nomme explicitement.

Ordre d'exécution (pipeline complet) :
    1. 1.exploration.ipynb
    2. 2.visualisation.ipynb
    3. 3.preprocessing_trend_group.ipynb
    4. 4_preprocessing_imputation.ipynb
    5. Boucle sur tous les notebooks "5.modelisation_*.ipynb"
    6. 6_synthese_modeles.ipynb

Les notebooks exécutés sont copiés (avec leurs outputs) 
dans notebooks/_executed/, SANS modifier les fichiers originaux.

Usage :
    python run_pipeline.py
    python run_pipeline.py --notebook 5.modelisation_light_gbm.ipynb
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

NOTEBOOKS_DIR = Path("notebooks")
OUTPUT_DIR = NOTEBOOKS_DIR / "_executed"
TIMEOUT = 1800  # secondes, par notebook

# --- Notebooks exécutés dans un ordre strict, imposé -----------------------
FIXED_ORDER = [
    "1.exploration.ipynb",
    "2.visualisation.ipynb",
    "3.preprocessing_trend_group.ipynb",
    "4_preprocessing_imputation.ipynb",
]
SYNTHESE_NOTEBOOK = "6_synthese_modeles.ipynb"


def find_notebooks() -> List[Path]:
    """Construit la liste ordonnée des notebooks du pipeline complet."""
    notebooks: List[Path] = []

    for name in FIXED_ORDER:
        path = NOTEBOOKS_DIR / name
        if not path.exists():
            print(f"[ERREUR] Notebook introuvable : {path}")
            sys.exit(1)
        notebooks.append(path)

    modelisation = sorted(NOTEBOOKS_DIR.glob("5.modelisation_*.ipynb"))
    if not modelisation:
        print(f"[ATTENTION] Aucun notebook '5.modelisation_*.ipynb' trouvé dans {NOTEBOOKS_DIR}")
    notebooks.extend(modelisation)

    path = NOTEBOOKS_DIR / SYNTHESE_NOTEBOOK
    if not path.exists():
        print(f"[ERREUR] Notebook introuvable : {path}")
        sys.exit(1)
    notebooks.append(path)

    return notebooks


def build_command(path: Path) -> List[str]:
    """Construit la commande jupyter nbconvert pour exécuter un notebook."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={TIMEOUT}",
        "--output-dir", str(OUTPUT_DIR),
        "--output", path.name,
        str(path),
    ]


def run_notebook(path: Path) -> float:
    """Exécute un notebook via nbconvert. Lève une RuntimeError si ça échoue."""
    cmd = build_command(path)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"Échec de l'exécution de {path.name} (code retour {result.returncode})")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Exécute le pipeline de notebooks, ou un seul notebook nommé.")
    parser.add_argument(
        "--notebook", type=str, default=None,
        help="Nom d'un notebook à exécuter seul (ex. --notebook 5.modelisation_light_gbm.ipynb). "
             "Si fourni, le pipeline complet n'est pas exécuté.",
    )
    args = parser.parse_args()

    if args.notebook:
        path = NOTEBOOKS_DIR / args.notebook
        if not path.exists():
            print(f"[ERREUR] Notebook introuvable : {path}")
            sys.exit(1)
        notebooks = [path]
    else:
        notebooks = find_notebooks()

    print("\nPipeline à exécuter :")
    for i, nb in enumerate(notebooks, 1):
        print(f"  {i}. {nb.name}")
    print(f"\nCopie vers : {OUTPUT_DIR}\n")

    results = []
    global_start = time.time()
    for nb in notebooks:
        print(f"▶ {nb.name} ...", flush=True)
        try:
            elapsed = run_notebook(nb)
        except RuntimeError as e:
            print(f"❌ {e}")
            sys.exit(1)
        print(f"✅ {nb.name} ({elapsed:.1f}s)\n")
        results.append((nb.name, elapsed))

    total = time.time() - global_start
    print("\n" + "=" * 55)
    print("Résumé de l'exécution :")
    for name, elapsed in results:
        print(f"  {name:<45} {elapsed:>7.1f}s")
    print(f"\nTemps total : {total:.1f}s ({total / 60:.1f} min)")
    print("🎉 Pipeline exécuté avec succès.")


if __name__ == "__main__":
    main()