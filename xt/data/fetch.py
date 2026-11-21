"""Fetch StatsBomb open data (https://github.com/statsbomb/open-data) with a local cache.

The raw JSON files are read straight from GitHub and cached under ``data/raw/statsbomb``
(same layout as the open-data repo), so every file is downloaded once. We do not go through
``statsbombpy`` here because it re-downloads on every call; it stays in the requirements for
quick ad-hoc exploration.

Typical use::

    from xt.data import load_competitions, load_matches, load_events

    comps = load_competitions()
    matches = load_matches(competition_id=43, season_id=106)  # FIFA World Cup 2022
    events = load_events(matches.match_id.iloc[0])           # list[dict], raw StatsBomb events
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Repo root = two levels above this file (xt/data/fetch.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = Path(os.environ.get("BXT_DATA_DIR", REPO_ROOT / "data")) / "raw" / "statsbomb"

_session = requests.Session()


def _fetch_json(rel_path: str, cache_dir: Path = DEFAULT_CACHE, retries: int = 4):
    """Return the parsed JSON at ``BASE_URL/rel_path``, using ``cache_dir/rel_path`` as a cache."""
    local = Path(cache_dir) / rel_path
    if local.exists():
        with open(local, encoding="utf-8") as f:
            return json.load(f)

    url = f"{BASE_URL}/{rel_path}"
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=60)
            if r.status_code == 404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            data = r.json()
            break
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(local)
    return data


def load_competitions(cache_dir: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """All competition/season pairs available in the open data."""
    return pd.DataFrame(_fetch_json("competitions.json", cache_dir))


def load_matches(competition_id: int, season_id: int, cache_dir: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Matches of one competition-season, with team names and scores flattened."""
    raw = _fetch_json(f"matches/{competition_id}/{season_id}.json", cache_dir)
    df = pd.json_normalize(raw, sep="_")
    keep = [
        "match_id", "match_date", "kick_off", "competition_competition_id", "competition_competition_name",
        "season_season_id", "season_season_name", "home_team_home_team_name", "away_team_away_team_name",
        "home_score", "away_score", "match_week", "competition_stage_name", "match_status_360",
    ]
    df = df[[c for c in keep if c in df.columns]].rename(columns={
        "competition_competition_id": "competition_id",
        "competition_competition_name": "competition_name",
        "season_season_id": "season_id",
        "season_season_name": "season_name",
        "home_team_home_team_name": "home_team",
        "away_team_away_team_name": "away_team",
        "competition_stage_name": "competition_stage",
    })
    return df.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def load_events(match_id: int, cache_dir: Path = DEFAULT_CACHE) -> list[dict]:
    """Raw StatsBomb events of one match (list of dicts, as in the open-data repo)."""
    return _fetch_json(f"events/{match_id}.json", cache_dir)


def load_360(match_id: int, cache_dir: Path = DEFAULT_CACHE) -> list[dict] | None:
    """360 freeze frames of one match, or ``None`` when the match has no 360 data."""
    try:
        return _fetch_json(f"three-sixty/{match_id}.json", cache_dir)
    except FileNotFoundError:
        return None


def download_events(match_ids: Iterable[int], cache_dir: Path = DEFAULT_CACHE, n_workers: int = 8,
                    progress: bool = True) -> list[int]:
    """Download (or confirm cached) event files for many matches in parallel.

    Returns the list of match ids that failed.
    """
    match_ids = list(match_ids)
    failed: list[int] = []
    it = None
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(total=len(match_ids), desc="events")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(load_events, m, cache_dir): m for m in match_ids}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:  # noqa: BLE001 - we report failures to the caller
                failed.append(futures[fut])
            if it is not None:
                it.update(1)
    if it is not None:
        it.close()
    return failed
