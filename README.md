# Bayesian xT : expected threat bayésien pour le football

Projet IFT6269 (UdeM, automne 2026). On part du modèle *expected threat* (xT, Singh 2019) estimé
sur les données ouvertes de [StatsBomb](https://github.com/statsbomb/open-data), et on étudie une
version bayésienne : priors pour les cases peu observées, modèle de transition par type d'action
(passes, conduites, dribbles), inférence MCMC, et un meilleur xG.

Structure reprise du template [udem-ift6758/project-template](https://github.com/udem-ift6758/project-template) :
un package Python installable (`bxt`) qui sert de « source unique » pour les données et les
fonctions, et des notebooks qui l'utilisent.

```
bxt/
  data/statsbomb.py      téléchargement StatsBomb open data + cache local (data/raw/)
  data/clean.py          événements bruts -> table d'actions propre (data/processed/*.parquet)
  features/grid.py       grille du terrain (16 x 12 par défaut), comptes par case
  visualizations/pitch.py  heatmaps sur un terrain (mplsoccer)
scripts/
  fetch_data.py          CLI pour télécharger / construire les tables d'actions
  build_eda_notebook.py  régénère le notebook EDA
notebooks/
  01_eda_statsbomb.ipynb EDA orientée xT
tests/                   tests unitaires (pytest)
data/                    données locales, ignorées par git
figures/
```

## Installation

Avec conda :

    conda env create --file environment.yml
    conda activate bxt-conda-env
    pip install -e .
    python -m ipykernel install --user --name=bxt-conda-env

Ou avec pip + virtualenv :

    python -m venv ~/bxt-venv && source ~/bxt-venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Données

Les données ne sont pas versionnées. Pour les télécharger :

    python scripts/fetch_data.py --list               # compétitions disponibles
    python scripts/fetch_data.py --preset eda         # Coupe du monde 2022 + Premier League 2015/16 (~1 Go, ~2 min)
    python scripts/fetch_data.py --preset big5_1516   # 5 grands championnats 2015/16 (~1 800 matchs, ~5 Go)

Chaque fichier JSON n'est téléchargé qu'une fois (cache dans `data/raw/statsbomb/`, même
arborescence que le repo open-data). La table d'actions nettoyée est écrite dans
`data/processed/actions_<competition_id>_<season_id>.parquet`. La variable d'environnement
`BXT_DATA_DIR` permet de mettre les données ailleurs.

En Python :

```python
from bxt.data import load_actions
actions = load_actions([(43, 106), (2, 27)])  # (competition_id, season_id)
```

Colonnes principales : `type` (Pass, Carry, Dribble, Shot, et les pertes de balle), `x, y, end_x,
end_y` (repère StatsBomb 120 x 80, chaque équipe attaque vers x = 120), `success`, `sub_type`,
`pass_height`, `under_pressure`, `shot_xg`, `is_goal`, `set_piece`. Détails dans
`bxt/data/clean.py`.

Nettoyage appliqué : tirs au but (période 5) retirés, événements sans position retirés,
coordonnées ramenées dans le terrain. Les passes au résultat `Unknown` / `Injury Clearance` sont
marquées `ambiguous_outcome`. Les buts contre son camp ne sont pas des tirs (≈ 4 % des buts).

## Tests

    pytest -q

Données : StatsBomb Open Data, utilisées selon leurs conditions (citer StatsBomb comme source).
