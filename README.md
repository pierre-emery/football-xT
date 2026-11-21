# Bayesian xT : expected threat bayésien pour le football
# TODO expliquer le projet un peu plus tard

## Installation

Avec conda :

    conda env create --file environment.yml
    conda activate xt-conda-env
    pip install -e .
    python -m ipykernel install --user --name=xt-conda-env

Ou avec pip + virtualenv :

    python -m venv ~/xt-venv && source ~/xt-venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Données

Les données ne sont pas versionnées. Pour les télécharger, exécuter
`notebooks/01_fetch_dataset.ipynb`. La liste `PAIRS` choisit les compétitions :

- `(43, 106)` : Coupe du monde 2022 (64 matchs)
- `(2, 27)` : Premier League 2015/16
- `(2, 27), (11, 27), (9, 27), (12, 27), (7, 27)` : 5 grands championnats 2015/16 (~1 800 matchs, ~5 Go)

`load_competitions()` liste toutes les compétitions disponibles.

Chaque fichier JSON n'est téléchargé qu'une fois (cache dans `data/raw/statsbomb/`).
La table d'actions nettoyée est écrite dans
`data/processed/actions_<competition_id>_<season_id>.parquet`. La variable
d'environnement `BXT_DATA_DIR` permet de mettre les données ailleurs.

Pour recharger la table déjà construite dans un autre notebook :
En Python :

```python
from xt.data import load_actions
actions = load_actions([(43, 106), (2, 27)])  # (competition_id, season_id)
```

### Structure du projet

```
xt/
  data/statsbomb.py      fetch les données
  data/clean.py          événements bruts -> table d'actions propre (data/processed/*.parquet)
  features/grid.py       grille du terrain (16 x 12 par défaut), comptes par case
  visualizations/pitch.py  heatmaps sur un terrain (mplsoccer)
notebooks/
  00_eda.ipynb           EDA 
  01_fetch_dataset.ipynb Pipeline qui télécharge le jeu de données
data/                    données locales
figures/
reports/                 Fichiers latex, Rapport en format ICML 2018 modifié et Poster
```