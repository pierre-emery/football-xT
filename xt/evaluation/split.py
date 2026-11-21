"""Train / validation / test split by match, fixed once for the whole project.

Protocol (against data snooping):
* train / val / test: the 2015/16 leagues, split at random by match within each league.
* recent_val: recent tournament(s) to explore recent football and validate (World Cup 2022).
* recent_test: final recent test set (Euro 2024), LOCKED until the final evaluation.
* test is also used only once, at the end.
EDA and model development only use EDA_SPLITS. The split is built from match metadata only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.clean import PROCESSED_DIR
from ..data.fetch import load_matches

SPLIT_PATH = PROCESSED_DIR / "split.csv"
EDA_SPLITS = ("train", "val", "recent_val")
LOCKED_SPLITS = ("test", "recent_test")


def make_split(train_pairs, recent_val_pairs=((43, 106),), recent_test_pairs=((55, 282),),
               val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 6269) -> pd.DataFrame:
    """One row per match: match_id, competition_name, season_name, split."""
    rng = np.random.default_rng(seed)
    parts = []
    for pairs, kind in [(train_pairs, "train"), (recent_val_pairs, "recent_val"), (recent_test_pairs, "recent_test")]:
        for c, s in pairs:
            d = load_matches(c, s)[["match_id", "competition_name", "season_name"]]
            if kind == "train":
                d = d.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
                n = len(d)
                n_test, n_val = round(test_frac * n), round(val_frac * n)
                d["split"] = ["test"] * n_test + ["val"] * n_val + ["train"] * (n - n_test - n_val)
            else:
                d = d.assign(split=kind)
            parts.append(d)
    return pd.concat(parts, ignore_index=True).sort_values("match_id").reset_index(drop=True)


def eda_matches(split: pd.DataFrame, splits=EDA_SPLITS) -> set:
    """Match ids that EDA and model development are allowed to look at."""
    return set(split.loc[split["split"].isin(splits), "match_id"])


def save_split(split: pd.DataFrame, path=SPLIT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)


def load_split(path=SPLIT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)