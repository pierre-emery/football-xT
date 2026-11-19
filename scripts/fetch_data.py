"""Download StatsBomb open data and build the cleaned action tables.

Examples
--------
    python scripts/fetch_data.py --list                 # show available competitions/seasons
    python scripts/fetch_data.py --preset eda           # default EDA set (see PRESETS)
    python scripts/fetch_data.py -c 2 -s 27             # Premier League 2015/16
    python scripts/fetch_data.py -c 43 -s 106 --max-matches 10
"""
import argparse

import pandas as pd

from bxt.data import build_actions, load_competitions

# (competition_id, season_id)
PRESETS = {
    # Small and quick: World Cup 2022 (64 matches, 360 data available)
    "wc2022": [(43, 106)],
    # Full league seasons, 2015/16 "big five" (~1,800 matches): best for learning a dense xT grid
    "big5_1516": [(2, 27), (11, 27), (9, 27), (12, 27), (7, 27)],
    # What the EDA notebook uses by default
    "eda": [(43, 106), (2, 27)],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--competition", type=int)
    ap.add_argument("-s", "--season", type=int)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="list competitions and exit")
    args = ap.parse_args()

    if args.list:
        comps = load_competitions()
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(comps[["competition_id", "season_id", "competition_name", "season_name",
                         "competition_gender", "match_available_360"]].to_string(index=False))
        return

    if args.preset:
        pairs = PRESETS[args.preset]
    elif args.competition is not None and args.season is not None:
        pairs = [(args.competition, args.season)]
    else:
        ap.error("give --preset, or both --competition and --season (or --list)")

    for c, s in pairs:
        df = build_actions(c, s, max_matches=args.max_matches, n_workers=args.workers)
        print(f"competition {c} season {s}: {df.match_id.nunique()} matches, {len(df):,} actions")


if __name__ == "__main__":
    main()
